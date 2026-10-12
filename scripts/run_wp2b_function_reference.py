"""WP2b - function expert: Doppler timing measures in healthy fetuses and a healthy reference.

Data: the 60 NInFEA strips (39 women, weeks 21 to 27, all fetuses healthy). The display
orientation of each strip is taken from run ``wp3_trust_check``. The detector and the choice of
measures were fixed on the 19 annotated strips that are not NInFEA recordings.

Reported:
  how often the measures can be read, and their values
  repeatability: first half against second half of a recording; recording against recording of
    the same woman (intraclass correlation)
  a conformal p-value against the healthy reference, leave-one-woman-out, and how often a
    healthy recording is flagged
  how large a change of each interval would be flagged (a measured healthy value plus a stated
    shift - labelled as a simulation, not disease)

No abnormal Doppler recording exists in the data, so no sensitivity is reported.

    python scripts/run_wp2b_function_reference.py
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.data.ninfea import ALL_RECORDS, strip_file, woman_of_record  # noqa: E402
from sfr.data.pwd import load_strip  # noqa: E402
from sfr.doppler.cycles import CycleParams, detect_cycles, prepare  # noqa: E402
from sfr.doppler.envelope import extract_envelopes  # noqa: E402
from sfr.doppler.measures import MIN_CYCLES, TEMPLATE_MEASURES, build_template, cycle_similarity, template_measures  # noqa: E402
from sfr.doppler.reference import (REFERENCE_MEASURES, conformal_p, fit_reference, intraclass_correlation,  # noqa: E402
                                   leave_one_woman_out, nonconformity, z_scores)
from sfr.eval.stats import bootstrap_ci, clopper_pearson  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

TRUST_RUN = "wp3_trust_check"
CYCLE_FROZEN = CycleParams()
ALPHAS = (0.05, 0.10, 0.20)
SHIFTS_MS = tuple(range(5, 125, 5))
SEED = 0


def main() -> None:
    load_run(TRUST_RUN)
    with open(RUNS / TRUST_RUN / "per_recording.csv", encoding="utf-8") as handle:
        trust = {int(r["record"]): r for r in csv.DictReader(handle)}
    who = woman_of_record()

    started = time.time()
    rows, halves = [], []
    for number in ALL_RECORDS:
        envelopes = extract_envelopes(load_strip(strip_file(number)))
        prepared = prepare(envelopes, params=CYCLE_FROZEN)
        side = trust[number]["side_used"]
        cycles = detect_cycles(envelopes, side, params=CYCLE_FROZEN, prepared=prepared)
        template = build_template(cycles, prepared)
        measures = template_measures(template)
        _, similarity = cycle_similarity(template)
        n_intervals = max(cycles.n - 1, 0)
        first = np.arange(n_intervals) < n_intervals // 2
        half_a = template_measures(build_template(cycles, prepared, keep=first))
        half_b = template_measures(build_template(cycles, prepared, keep=~first))
        rows.append({
            "record": number, **who[number], "ejection_side": side, "trusted": trust[number]["trusted"] == "True",
            "duration_s": envelopes.duration_s, "n_cycles": cycles.n, "n_template_cycles": template.n_cycles if template else 0,
            "clicks_seen_share": float(np.mean(cycles.has_click)) if cycles.n else 0.0, "track_score": cycles.track_score,
            "similarity_median": float(np.median(similarity)) if similarity.size else float("nan"),
            **measures,
        })
        halves.append((half_a, half_b))
        print(f"record {number} done ({time.time() - started:.0f} s)", flush=True)

    usable = [r for r in rows if all(np.isfinite(r[m]) for m in REFERENCE_MEASURES)]
    scored = leave_one_woman_out(usable)
    by_record = {s["record"]: s for s in scored}
    for r in rows:
        s = by_record.get(r["record"])
        r["score"] = s["score"] if s else float("nan")
        r["p_value"] = s["p_value"] if s else float("nan")
        for measure in REFERENCE_MEASURES:
            r[f"z_{measure}"] = s["z"][measure] if s else float("nan")

    # ---- how large a shift would be flagged (simulation on healthy values) -------------------------
    scores = np.array([s["score"] for s in scored])
    owner = np.array([s["woman"] for s in scored])
    shift_rows = []
    women = sorted({r["woman"] for r in usable})
    references = {w: fit_reference([r for r in usable if r["woman"] != w]) for w in women}
    interval_measures = [m for m in REFERENCE_MEASURES if m != "cycle_length_ms"]
    for measure in interval_measures:
        for sign in (1, -1):
            for shift in SHIFTS_MS:
                p_values = []
                for r in usable:
                    altered = dict(r)
                    altered[measure] = r[measure] + sign * shift
                    score = nonconformity(z_scores(altered, references[r["woman"]]))
                    p_values.append(conformal_p(score, scores[owner != r["woman"]]))
                p_values = np.array(p_values)
                shift_rows.append({"measure": measure, "shift_ms": sign * shift, "n": int(p_values.size),
                                   **{f"flagged_share_alpha_{a}": float(np.mean(p_values <= a)) for a in ALPHAS}})

    run = Run(
        "wp2b_function_reference",
        description="Doppler timing measures of the 60 NInFEA strips: repeatability, healthy reference with conformal p-values, detectable shifts",
        script="scripts/run_wp2b_function_reference.py",
        config={"cycles": asdict(CYCLE_FROZEN), "trust_run": TRUST_RUN, "measures": TEMPLATE_MEASURES,
                "reference_measures": REFERENCE_MEASURES, "min_cycles": MIN_CYCLES, "alphas": ALPHAS, "shifts_ms": SHIFTS_MS},
        inputs={"ninfea_strips": [strip_file(n) for n in ALL_RECORDS], "trust": [RUNS / TRUST_RUN / "per_recording.csv"]},
        n_real=len(ALL_RECORDS),
        n_synthetic=len(shift_rows) * len(usable),
        synthetic_kind="measured healthy value plus a stated shift of one interval (file detectable_shifts.csv); no image or signal is altered",
        seed=SEED,
    )
    run.write_csv("per_recording.csv", rows)
    run.write_csv("detectable_shifts.csv", shift_rows)

    # ---- values and how often they can be read -------------------------------------------------------
    values = {}
    for measure in TEMPLATE_MEASURES:
        x = np.array([r[measure] for r in rows], dtype=float)
        x = x[np.isfinite(x)]
        values[measure] = {"n_readable": int(x.size), "share_readable": x.size / len(rows),
                           "median": float(np.median(x)) if x.size else float("nan"),
                           "q25": float(np.percentile(x, 25)) if x.size else float("nan"),
                           "q75": float(np.percentile(x, 75)) if x.size else float("nan"),
                           "min": float(x.min()) if x.size else float("nan"), "max": float(x.max()) if x.size else float("nan")}

    # ---- repeatability ---------------------------------------------------------------------------------
    repeat = {}
    all_women = sorted({r["woman"] for r in rows})
    for measure in TEMPLATE_MEASURES:
        diff = np.array([a[measure] - b[measure] for a, b in halves], dtype=float)
        diff = diff[np.isfinite(diff)]
        groups = [np.array([r[measure] for r in rows if r["woman"] == w], dtype=float) for w in all_women]
        repeated = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() >= 2]
        icc = intraclass_correlation(repeated)
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(2000):
            chosen = rng.integers(0, len(repeated), len(repeated))
            value = intraclass_correlation([repeated[i] for i in chosen])
            if np.isfinite(value):
                draws.append(value)
        within = np.sqrt(np.mean([np.var(g, ddof=1) for g in repeated])) if repeated else float("nan")
        repeat[measure] = {
            "split_half_n": int(diff.size), "split_half_abs_diff_median": float(np.median(np.abs(diff))) if diff.size else float("nan"),
            "split_half_sd_of_one_half": float(np.std(diff, ddof=1) / np.sqrt(2)) if diff.size > 1 else float("nan"),
            "women_with_repeats": len(repeated), "icc": icc,
            "icc_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else [float("nan")] * 2,
            "within_woman_sd": float(within),
        }

    # ---- healthy recordings flagged ----------------------------------------------------------------------
    p = np.array([s["p_value"] for s in scored])
    woman_of = np.array([s["woman"] for s in scored])
    flagged = {}
    for alpha in ALPHAS:
        hits = p <= alpha
        share = bootstrap_ci(np.arange(len(women)), lambda index: float(np.mean(np.concatenate(
            [hits[woman_of == women[int(i)]] for i in index]))), seed=SEED)
        women_hit = sum(bool(hits[woman_of == w].any()) for w in women)
        flagged[str(alpha)] = {
            "recordings": int(hits.size), "recordings_flagged": int(hits.sum()), "share": float(hits.mean()),
            "share_ci_by_woman": [share[1], share[2]],
            "women": len(women), "women_with_a_flagged_recording": int(women_hit),
            "women_share_ci": list(clopper_pearson(women_hit, len(women))),
        }
    trusted_only = [r for r in usable if r["trusted"]]
    scored_trusted = leave_one_woman_out(trusted_only) if len({r["woman"] for r in trusted_only}) > 5 else []
    p_trusted = np.array([s["p_value"] for s in scored_trusted])

    detect = {}
    for measure in interval_measures:
        detect[measure] = {}
        for alpha in ALPHAS[:2]:
            for sign, label in ((1, "longer"), (-1, "shorter")):
                chosen = sorted([s for s in shift_rows if s["measure"] == measure and np.sign(s["shift_ms"]) == sign],
                                key=lambda s: abs(s["shift_ms"]))
                at = {}
                for level in (0.5, 0.8):
                    hit = [abs(s["shift_ms"]) for s in chosen if s[f"flagged_share_alpha_{alpha}"] >= level]
                    at[f"smallest_shift_ms_for_{int(level * 100)}pct"] = hit[0] if hit else None
                detect[measure][f"{label}_alpha_{alpha}"] = at

    reference_all = fit_reference(usable)
    metrics = {
        "n_recordings": len(rows), "n_women": len(all_women), "usable_recordings": len(usable),
        "usable_women": len(women), "not_usable": [r["record"] for r in rows if r not in usable],
        "ejection_side": {"upper": sum(r["ejection_side"] == "upper" for r in rows), "lower": sum(r["ejection_side"] == "lower" for r in rows)},
        "values": values, "repeatability": repeat, "reference_lines_all_usable": reference_all,
        "healthy_recordings_flagged": flagged,
        "healthy_recordings_flagged_trust_check_passed_only": {
            str(a): {"recordings": int(p_trusted.size), "flagged": int((p_trusted <= a).sum())} for a in ALPHAS} if p_trusted.size else {},
        "smallest_p_value_possible": float(1.0 / (min(np.sum(woman_of != w) for w in women) + 1)),
        "detectable_shift": detect,
        "statement": "All fetuses are healthy. The numbers describe false alarms and repeatability only; no sensitivity is measured.",
    }
    run.finish(metrics)

    print("\nMeasure                     readable   median [q25, q75]      split-half |d|   ICC [95% CI] (women)   within-woman SD")
    for measure in TEMPLATE_MEASURES:
        v, q = values[measure], repeat[measure]
        print(f"{measure:26s} {v['n_readable']:2d}/{len(rows)}   {v['median']:7.2f} [{v['q25']:.2f}, {v['q75']:.2f}]   {q['split_half_abs_diff_median']:8.2f}      "
              f"{q['icc']:.2f} [{q['icc_ci'][0]:.2f}, {q['icc_ci'][1]:.2f}] ({q['women_with_repeats']})   {q['within_woman_sd']:.2f}")
    print("usable recordings:", len(usable), "of", len(rows), "; women:", len(women), "; not usable:", metrics["not_usable"])
    print("healthy recordings flagged:", {a: (f["recordings_flagged"], f["recordings"], round(f["share"], 3), np.round(f["share_ci_by_woman"], 3).tolist()) for a, f in flagged.items()})
    print("trust-check-passed only:", metrics["healthy_recordings_flagged_trust_check_passed_only"])
    print("smallest possible p:", round(metrics["smallest_p_value_possible"], 4))
    for measure, d in detect.items():
        print("detectable shift", measure, d)


if __name__ == "__main__":
    main()
