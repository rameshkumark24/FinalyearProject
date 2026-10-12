"""WP2a - cardiac-cycle detection on pulsed-wave Doppler strips against cardiologist marks.

Data: the 43 annotated strips of Sulas et al. (Mendeley Data, doi:10.17632/pg2djw3sjc.1).
Settings were tuned on the 19 development strips only (``splits/pwd_cycle_marks.json``). This
script runs the frozen detector and four ablations on all strips and reports the two splits
separately. The constant shift between a detected closure and a mark is calibrated on the
development strips of each configuration and then applied unchanged to the test strips.

    python scripts/run_wp2_doppler_cycles.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.pwd import annotated_ids, annotated_path, load_cycle_marks, load_strip  # noqa: E402
from sfr.doppler.cycles import CycleParams, detect_cycles, prepare  # noqa: E402
from sfr.doppler.envelope import PIXELS_PER_SECOND, extract_envelopes  # noqa: E402
from sfr.doppler.marks import rates_in_marked_spans, score_starts, tiled  # noqa: E402
from sfr.doppler.measures import build_template, cycle_similarity  # noqa: E402
from sfr.doppler.orientation import from_doppler  # noqa: E402
from sfr.ecg.metrics import match_pairs  # noqa: E402
from sfr.eval.stats import auroc, bootstrap_ci, clopper_pearson  # noqa: E402
from sfr.paths import MENDELEY_PWD_DIR, SPLITS  # noqa: E402
from sfr.provenance import Run  # noqa: E402

FROZEN = CycleParams()
CONFIGS = {
    "full": FROZEN,
    "wave_end_start": replace(FROZEN, use_click_start=False),
    "no_click_removal": replace(FROZEN, use_click_removal=False),
    "no_floor": replace(FROZEN, use_floor=False),
    "no_tracking": replace(FROZEN, use_tracker=False),
}
TOLERANCE, WIDE_TOLERANCE = 15, 25          # columns: 53 ms and 88 ms at 284 columns per second
RATE_LIMIT_BPM = 5.0


def calibrate_offset(cycles: dict, marks: dict, numbers: list[int]) -> float:
    residual: list[float] = []
    for n in numbers:
        starts = np.sort(cycles[n].starts)
        residual += [marks[n][j] - starts[i] for i, j in match_pairs(starts, marks[n], 40)]
    return float(np.median(residual)) if residual else 0.0


def summarise(rows: list[dict]) -> dict:
    f1 = np.array([r["f1"] for r in rows])
    wide = np.array([r["f1_wide"] for r in rows])
    total_marks = sum(r["n_marks"] for r in rows)
    total_hand = sum(r["n_hand_placed"] for r in rows)
    with_rate = [r for r in rows if np.isfinite(r["marked_bpm"]) and np.isfinite(r["detected_bpm"])]
    n_rate_ok = sum(abs(r["detected_bpm"] - r["marked_bpm"]) <= RATE_LIMIT_BPM for r in with_rate)
    mean, median, mean_wide = bootstrap_ci(f1, np.mean), bootstrap_ci(f1, np.median), bootstrap_ci(wide, np.mean)
    recall = bootstrap_ci(np.arange(len(rows)), lambda index: sum(rows[int(i)]["tp"] for i in index)
                          / max(1, sum(rows[int(i)]["n_marks"] for i in index)))
    return {
        "n_strips": len(rows), "n_marks": total_marks,
        "f1_mean": mean[0], "f1_mean_ci": [mean[1], mean[2]],
        "f1_median": median[0], "f1_median_ci": [median[1], median[2]],
        "f1_min": float(f1.min()), "share_f1_at_least_0.9": float(np.mean(f1 >= 0.9)),
        "f1_wide_mean": mean_wide[0], "f1_wide_mean_ci": [mean_wide[1], mean_wide[2]],
        "pooled_recall": recall[0], "pooled_recall_ci": [recall[1], recall[2]],
        "pooled_recall_hand_placed": sum(r["tp_hand_placed"] for r in rows) / total_hand if total_hand else float("nan"),
        "pooled_precision_dense": sum(r["tp"] for r in rows) / max(1, sum(r["tp"] + r["fp"] for r in rows)),
        "rate_n_strips": len(with_rate), "rate_within_limit_n": int(n_rate_ok),
        "rate_within_limit_share": n_rate_ok / len(with_rate) if with_rate else float("nan"),
        "rate_within_limit_ci": list(clopper_pearson(n_rate_ok, len(with_rate))) if with_rate else [float("nan")] * 2,
        "rate_abs_error_median_bpm": float(np.median([abs(r["detected_bpm"] - r["marked_bpm"]) for r in with_rate])) if with_rate else float("nan"),
        "rate_abs_error_max_bpm": float(np.max([abs(r["detected_bpm"] - r["marked_bpm"]) for r in with_rate])) if with_rate else float("nan"),
    }


def main() -> None:
    split = json.loads((SPLITS / "pwd_cycle_marks.json").read_text(encoding="utf-8"))
    overlap = json.loads((SPLITS / "pwd_overlap.json").read_text(encoding="utf-8"))
    woman_of = {s["img"]: s["woman"] for s in overlap["strips"]}
    development, test = split["development"], split["test"]
    numbers = annotated_ids()
    assert sorted(development + test) == numbers and not set(development) & set(test)
    marks = load_cycle_marks()

    run = Run(
        "wp2_doppler_cycles",
        description="Cardiac-cycle start detection on 43 annotated fetal Doppler strips: frozen detector and four ablations",
        script="scripts/run_wp2_doppler_cycles.py",
        config={"frozen": asdict(FROZEN), "configs": {k: asdict(v) for k, v in CONFIGS.items()},
                "tolerance_columns": TOLERANCE, "wide_tolerance_columns": WIDE_TOLERANCE,
                "columns_per_second": PIXELS_PER_SECOND, "rate_limit_bpm": RATE_LIMIT_BPM,
                "split_file": "splits/pwd_cycle_marks.json", "ejection_side": "lower (all annotated strips; Sulas 2021)"},
        inputs={"annotated_strips": [annotated_path(n) for n in numbers] + [MENDELEY_PWD_DIR / "annotations.mat"]},
        n_real=len(numbers),
    )

    started = time.time()
    envelopes = {}
    for n in numbers:
        envelopes[n] = extract_envelopes(load_strip(annotated_path(n)))
    print(f"envelopes of {len(numbers)} strips in {time.time() - started:.0f} s", flush=True)

    rows: list[dict] = []
    offsets: dict[str, float] = {}
    orientation_rows: list[dict] = []
    for config_name, params in CONFIGS.items():
        prepared = {n: prepare(envelopes[n], params=params) for n in numbers}
        cycles = {n: detect_cycles(envelopes[n], "lower", params=params, prepared=prepared[n]) for n in numbers}
        offset = calibrate_offset(cycles, marks, development)
        offsets[config_name] = offset
        for n in numbers:
            strict = score_starts(cycles[n].starts, marks[n], TOLERANCE, offset)
            wide = score_starts(cycles[n].starts, marks[n], WIDE_TOLERANCE, offset)
            marked_bpm, detected_bpm, n_gaps = rates_in_marked_spans(cycles[n].starts, marks[n], PIXELS_PER_SECOND, offset)
            hand = ~tiled(marks[n])
            matched = set(strict["matched_mark_index"])
            row = {
                "img": n, "split": "development" if n in development else "test", "woman": woman_of[n] or "",
                "config": config_name, "width": int(envelopes[n].width), "duration_s": envelopes[n].duration_s,
                "n_marks": strict["n_marks"], "n_hand_placed": int(hand.sum()),
                "tp_hand_placed": int(sum(1 for j in np.flatnonzero(hand) if j in matched)),
                "n_cycles": cycles[n].n, "tp": strict["tp"], "fp": strict["fp"], "fn": strict["fn"],
                "recall": strict["recall"], "precision": strict["precision"], "f1": strict["f1"],
                "f1_wide": wide["f1"], "recall_wide": wide["recall"],
                "residual_median": strict["residual_median"], "residual_iqr": strict["residual_iqr"],
                "marked_bpm": marked_bpm, "detected_bpm": detected_bpm, "n_rate_gaps": n_gaps,
                "clicks_seen_share": float(np.mean(cycles[n].has_click)) if cycles[n].n else 0.0,
                "track_score": cycles[n].track_score,
            }
            if config_name == "full":
                found = from_doppler(prepared[n], params=params)
                orientation_rows.append({"img": n, "split": row["split"], "inferred_ejection_side": found["side"],
                                         "correct": found["side"] == "lower", "rule": found["rule"],
                                         "log_area_ratio": found["log_area_ratio"], "lag": found["lag"]})
                template = build_template(cycles[n], prepared[n])
                index, similarity = cycle_similarity(template)
                near = {i for i, _ in match_pairs(np.sort(cycles[n].starts) + offset, marks[n], TOLERANCE)}
                label = np.array([1 if i in near else 0 for i in index])
                enough = label.size and min(label.sum(), label.size - label.sum()) >= 5
                row["similarity_auroc_vs_marked"] = auroc(label, similarity) if enough else float("nan")
                row["n_cycles_marked"], row["n_cycles_unmarked"] = int(label.sum()), int(label.size - label.sum())
            rows.append(row)
        print(f"{config_name} done ({time.time() - started:.0f} s)", flush=True)

    run.write_csv("per_strip.csv", rows)
    run.write_csv("orientation_doppler_only.csv", orientation_rows)

    metrics: dict = {"tolerance_columns": TOLERANCE, "tolerance_ms": 1000 * TOLERANCE / PIXELS_PER_SECOND,
                     "wide_tolerance_columns": WIDE_TOLERANCE, "mark_offset_columns": offsets, "by_config": {}}
    for config_name in CONFIGS:
        selected = [r for r in rows if r["config"] == config_name]
        metrics["by_config"][config_name] = {
            "development": summarise([r for r in selected if r["split"] == "development"]),
            "test": summarise([r for r in selected if r["split"] == "test"]),
        }
    full_test = [r for r in rows if r["config"] == "full" and r["split"] == "test"]
    by_woman: dict[str, list[float]] = {}
    for r in full_test:
        by_woman.setdefault(r["woman"], []).append(r["f1"])
    woman_f1 = np.array([np.mean(v) for v in by_woman.values()])
    woman_ci = bootstrap_ci(woman_f1, np.mean)
    metrics["test_by_woman"] = {"n_women": int(woman_f1.size), "f1_mean": woman_ci[0], "f1_mean_ci": [woman_ci[1], woman_ci[2]],
                                "f1_min": float(woman_f1.min())}
    orientation: dict = {}
    for name in ("development", "test"):
        chosen = [r for r in orientation_rows if r["split"] == name]
        correct = sum(r["correct"] for r in chosen)
        orientation[name] = {"n": len(chosen), "correct": int(correct), "share": correct / len(chosen),
                             "ci": list(clopper_pearson(correct, len(chosen))),
                             "wrong": [r["img"] for r in chosen if not r["correct"]]}
    metrics["orientation_doppler_only"] = orientation
    for name in ("development", "test"):
        values = np.array([r["similarity_auroc_vs_marked"] for r in rows
                           if r["config"] == "full" and r["split"] == name and np.isfinite(r.get("similarity_auroc_vs_marked", np.nan))])
        metrics.setdefault("similarity_vs_marked", {})[name] = {
            "n_strips": int(values.size), "auroc_median": float(np.median(values)) if values.size else float("nan"),
            "auroc_min": float(values.min()) if values.size else float("nan"),
            "auroc_max": float(values.max()) if values.size else float("nan")}

    result = metrics["by_config"]["full"]["test"]
    metrics["gate_G2"] = {
        "rule": "on the held-out strips: median F1 >= 0.90 at 15 columns, and Doppler rate within 5 bpm of the marked rate in >= 90% of strips",
        "median_f1_test": result["f1_median"], "rate_within_limit_share_test": result["rate_within_limit_share"],
        "passed": bool(result["f1_median"] >= 0.90 and result["rate_within_limit_share"] >= 0.90),
    }
    run.finish(metrics)

    print("\nConfig            split   n  mean F1@15 [95% CI]      median  min    F1@25   recall  hand   rate ok")
    for config_name in CONFIGS:
        for name in ("development", "test"):
            m = metrics["by_config"][config_name][name]
            print(f"{config_name:17s} {name[:5]:5s} {m['n_strips']:3d}  {m['f1_mean']:.3f} [{m['f1_mean_ci'][0]:.3f}, {m['f1_mean_ci'][1]:.3f}]"
                  f"   {m['f1_median']:.3f}  {m['f1_min']:.3f}  {m['f1_wide_mean']:.3f}   {m['pooled_recall']:.3f}  "
                  f"{m['pooled_recall_hand_placed']:.3f}  {m['rate_within_limit_n']}/{m['rate_n_strips']}")
    print("\nMark offsets (columns):", offsets)
    print("Test by woman:", metrics["test_by_woman"])
    print("Orientation from Doppler alone:", metrics["orientation_doppler_only"])
    print("Similarity vs marked:", metrics["similarity_vs_marked"])
    print("Gate G2:", metrics["gate_G2"])


if __name__ == "__main__":
    main()
