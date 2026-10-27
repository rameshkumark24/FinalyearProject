"""Build the results summary, the tables and the figures from the run folders.

    python scripts/build_report.py

Nothing in this file is a result. Every number that appears in ``reports/`` is read from
``runs/<run>/metrics.json`` or from a per-case file of that run; ``python -m sfr.audit`` checks
that this file holds no typed-in metric values. ``reports/manifest.json`` lists, for every
output, the runs it was built from.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import REPORTS, RUNS  # noqa: E402
from sfr.provenance import load_run  # noqa: E402

TABLES, FIGURES = REPORTS / "tables", REPORTS / "figures"
HALF = 1 / 2
THIRD = 1 / 3
BOUNDS_RUN, TRADEOFF_RUN, CHECKS_RUN = "wp7_dependence_bounds", "wp8_tradeoff", "wp9_checks_after_results"
MANIFEST: list[dict] = []
GATE_RULES = {
    "G4": "every row has n, an interval and a run folder; the checkpoint hash matches the published file; the negative-control rows are present",
    "G5": "the released reference model reproduced; pooled patient-level AUROC with an interval; shortcut audit reported",
    "G6": "all property tests pass; every table row states how many scores are real, how many virtual, and which coupling",
}


# ---- helpers -------------------------------------------------------------------------------------------

def read_csv(run: str, name: str) -> list[dict]:
    with open(RUNS / run / name, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def interval(pair, digits: int = 3) -> str:
    return f"[{num(pair[0], digits)}, {num(pair[1], digits)}]"


def pct(value, digits: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * float(value):.{digits}f}%"


def table(name: str, header: list[str], rows: list[list], runs: list[str]) -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    MANIFEST.append({"file": f"tables/{name}.csv", "runs": runs})
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def save(figure, name: str, runs: list[str]) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(figure)
    MANIFEST.append({"file": f"figures/{name}.png", "runs": runs})
    return f"![{name}](figures/{name}.png)"


def state(passed) -> str:
    return "met" if passed else "NOT met"


# ---- sections ------------------------------------------------------------------------------------------

def section_wp1() -> str:
    run = "wp1_cinc2013"
    metrics, _ = load_run(run)
    rows = []
    for config, subsets in metrics["by_config"].items():
        for subset in ("development", "test", "test_without_doubted_references"):
            m = subsets[subset]
            rows.append([config, subset.replace("_", " "), m["n_recordings"], f"{num(m['f1_mean'])} {interval(m['f1_mean_ci'])}",
                         num(m["f1_median"]), pct(m["share_f1_at_least_0.9"], 0),
                         f"{m['follows_mother_n']}/{m['n_recordings']}", num(m["hr_abs_error_median_bpm"], 2)])
    text = table("t1_fetal_beat_detection", ["Pipeline", "Split", "n", "Mean F1 [95% CI]", "Median F1", "F1 >= 0.9",
                                             "Follows the mother", "Median rate error (bpm)"], rows, [run])
    per_record = read_csv(run, "per_record.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    configs = list(metrics["by_config"])
    for position, config in enumerate(configs):
        values = np.array([float(r["f1"]) for r in per_record if r["config"] == config and r["split"] == "test"])
        jitter = np.random.default_rng(position).uniform(-THIRD / 2, THIRD / 2, values.size)
        axis.scatter(position + jitter, values, s=12, alpha=HALF)
        axis.hlines(np.median(values), position - THIRD, position + THIRD, color="black")
    axis.set_xticks(range(len(configs)), [c.replace("_", "\n") for c in configs])
    axis.set_ylabel("Fetal beat F1 per recording (held-out)")
    axis.set_title("CinC 2013 set A, test recordings: full pipeline and ablations (bar = median)")
    picture = save(figure, "f1_fetal_beat_detection", [run])
    gate = metrics["gate_G1"]
    lead = (f"Gate G1 ({gate['rule']}): **{state(gate['passed'])}** - median F1 {num(gate['median_f1_test'])}, "
            f"follows-the-mother rate {pct(gate['follows_mother_rate_test'])}.")
    return "\n\n".join(["## 1. Fetal beat detection (CinC 2013 set A)", lead, text, picture])


def section_wp2() -> str:
    run = "wp2_doppler_cycles"
    metrics, _ = load_run(run)
    rows = []
    for config, splits in metrics["by_config"].items():
        for split in ("development", "test"):
            m = splits[split]
            rows.append([config, split, m["n_strips"], m["n_marks"], f"{num(m['f1_mean'])} {interval(m['f1_mean_ci'])}",
                         num(m["f1_median"]), num(m["f1_min"]), num(m["f1_wide_mean"]), num(m["pooled_recall"]),
                         num(m["pooled_recall_hand_placed"]), f"{m['rate_within_limit_n']}/{m['rate_n_strips']}"])
    text = table("t2_doppler_cycle_detection",
                 ["Detector", "Split", "Strips", "Marks", "Mean F1 at 15 columns [95% CI]", "Median", "Lowest strip",
                  "Mean F1 at 25 columns", "Recall", "Recall, hand-placed marks", "Rate within 5 bpm"], rows, [run])
    woman = metrics["test_by_woman"]
    orientation = metrics["orientation_doppler_only"]
    similarity = metrics["similarity_vs_marked"]["test"]
    gate = metrics["gate_G2"]
    notes = [
        f"Gate G2 ({gate['rule']}): **{state(gate['passed'])}** - median F1 {num(gate['median_f1_test'])}, "
        f"rate within the limit in {pct(gate['rate_within_limit_share_test'], 0)} of held-out strips.",
        f"Held-out strips grouped by woman ({woman['n_women']} women): mean F1 {num(woman['f1_mean'])} {interval(woman['f1_mean_ci'])}, lowest {num(woman['f1_min'])}.",
        f"Display orientation from the Doppler alone: {orientation['test']['correct']}/{orientation['test']['n']} held-out strips correct "
        f"(95% CI {interval(orientation['test']['ci'], 2)}; wrong: {orientation['test']['wrong']}); development {orientation['development']['correct']}/{orientation['development']['n']}.",
        f"Agreement of a per-cycle similarity index with the cardiologist's choice of measurable cycles (held-out strips with at least five cycles of each kind, "
        f"n = {similarity['n_strips']}): median AUROC {num(similarity['auroc_median'], 2)}, range {num(similarity['auroc_min'], 2)} to {num(similarity['auroc_max'], 2)} - weak.",
    ]
    per_strip = read_csv(run, "per_strip.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    configs = list(metrics["by_config"])
    for position, config in enumerate(configs):
        for shift, split, colour in ((-THIRD / 2, "development", "tab:grey"), (THIRD / 2, "test", "tab:blue")):
            values = np.array([float(r["f1"]) for r in per_strip if r["config"] == config and r["split"] == split])
            axis.scatter(np.full(values.size, position + shift), values, s=14, alpha=HALF, color=colour,
                         label=split if position == 0 else None)
    axis.set_xticks(range(len(configs)), [c.replace("_", "\n") for c in configs])
    axis.set_ylabel("Cycle-start F1 per strip (15 columns)")
    axis.set_title("Doppler cycle detection against cardiologist marks")
    axis.legend()
    picture = save(figure, "f2_doppler_cycle_detection", [run])
    return "\n\n".join(["## 2. Cardiac cycles on Doppler strips (annotated set)", *notes, text, picture])


def section_wp3() -> str:
    run, first = "wp3_trust_check", "wp3_trust_check_v1_prespecified"
    metrics, _ = load_run(run)
    first_metrics, _ = load_run(first)
    rows = []
    for label, source in (("first held-out run (as pre-specified)", first_metrics), ("revised extractor", metrics)):
        for split in ("development", "test"):
            m = source["by_split"][split]
            rate = m["rate_recording_level"]
            rows.append([label, split, m["n_recordings"], m["n_women"], f"{m['trusted_recordings']}/{m['n_recordings']}",
                         f"{num(m['auroc_vs_maternal']['auroc'])} {interval(m['auroc_vs_maternal']['ci'])}",
                         f"{num(m['auroc_vs_swap']['auroc'])} {interval(m['auroc_vs_swap']['ci'])}",
                         f"{m['maternal_flagged']['flagged']}/{m['maternal_flagged']['n']}",
                         f"{m['swap_flagged']['flagged']}/{m['swap_flagged']['n']}",
                         interval([rate["all"]["loa_low"], rate["all"]["loa_high"]], 1),
                         interval([rate["trusted_only"]["loa_low"], rate["trusted_only"]["loa_high"]], 1)])
    text = table("t3_trust_check",
                 ["Version", "Split", "Recordings", "Women", "Pass the check", "AUROC vs mother's beats [95% CI]",
                  "AUROC vs another woman's ECG [95% CI]", "Mother's beats flagged", "Swaps flagged",
                  "Rate limits, all (bpm)", "Rate limits, passing (bpm)"], rows, [run, first])
    test = metrics["by_split"]["test"]
    gap_rows = [[key, value["flagged"], value["n"], pct(value["share"])] for key, value in test["swap_flagged"]["by_rate_gap_bpm"].items()]
    gap_text = table("t3b_swap_detection_by_rate_gap", ["Rate difference between the two fetuses (bpm)", "Flagged", "Pairs", "Share"],
                     gap_rows, [run])
    perturb_rows = []
    for kind, levels in test["perturbations_from_trusted_pairs"].items():
        for level, value in levels.items():
            perturb_rows.append([kind, level, value["n"], pct(value["flagged_share"]), num(value["locked_share_median"], 2)])
    perturb_text = table("t3c_trust_check_perturbations", ["Alteration of the fetal beat series", "Size", "Cases", "Flagged", "Median locked share"],
                         perturb_rows, [run])
    beat = test["beat_level_ecg_quality_passed_recordings"]
    gate = metrics["gate_G3"]
    ablation = test["cycle_start_ablation"]
    orientation = test["orientation"]
    notes = [
        f"Gate G3 ({gate['rule']}): **{state(gate['passed'])}**. Beat-to-beat limits of agreement {interval(gate['beat_level_limits_bpm'], 1)} bpm "
        f"(agreement part {state(gate['beat_level_passed'])}); AUROC for the mother's beats {num(gate['auroc_maternal_test'])} (detection part {state(gate['auroc_passed'])}).",
        f"Held-out recordings that pass the extractor's own quality check: {test['ecg_quality_passed_recordings']} of {test['n_recordings']}. "
        f"In them, {beat['n_pairs']} paired intervals: median absolute difference {num(beat['abs_diff_median_ms'], 1)} ms, "
        f"{pct(beat['share_within_10ms'])} within 10 ms, {pct(beat['share_within_20ms'])} within 20 ms. The differences have heavy tails, "
        f"so the classical limits are wide although most beats agree closely.",
        f"Recordings not passing the check (held-out): " + "; ".join(f"{x['record']} ({x['reasons']})" for x in test["not_trusted"]) + ".",
        f"Cycle start at the valve closure click against the end of the ejection wave: median locked share {num(ablation['closure_click']['locked_share_median'], 2)} "
        f"against {num(ablation['wave_end']['locked_share_median'], 2)}; the click is better in {ablation['recordings_where_click_is_better']} of {ablation['n']} held-out recordings.",
        f"R peak to valve closure in passing held-out recordings: median {num(test['delay_r_to_closure_ms_trusted']['median'], 0)} ms "
        f"(range {num(test['delay_r_to_closure_ms_trusted']['min'], 0)} to {num(test['delay_r_to_closure_ms_trusted']['max'], 0)} ms).",
        f"Display orientation where it is known (held-out, n = {orientation['known_n']}): from the ECG {orientation['from_ecg_correct']} correct, "
        f"from the Doppler alone {orientation['from_doppler_correct']} correct; the two methods agree on {orientation['unknown_two_methods_agree']} of {orientation['unknown_n']} strips of unknown orientation.",
    ]
    if (RUNS / CHECKS_RUN / "metrics.json").exists():
        robust = load_run(CHECKS_RUN)[0]["C_trust_check_robust_agreement"]
        limits = beat["rate_diff_percentiles_plausible_bpm"]
        classical = beat["rate_limits_plausible_pairs"]
        target = robust["gate_G3_target_bpm"]
        notes.append(
            f"Robust agreement, added after seeing the result (run {CHECKS_RUN}, recomputed from the beat times saved by this run): of the same "
            f"{robust['pairs']} beat pairs, the absolute interval difference has median {num(robust['abs_interval_difference_ms']['50'], 1)} ms, "
            f"90th percentile {num(robust['abs_interval_difference_ms']['90'], 1)} ms and 95th percentile {num(robust['abs_interval_difference_ms']['95'], 1)} ms. "
            f"On the {robust['plausible_pairs']} plausible pairs the absolute rate difference has median {num(robust['abs_rate_difference_bpm_plausible_pairs']['50'], 1)} bpm, "
            f"90th percentile {num(robust['abs_rate_difference_bpm_plausible_pairs']['90'], 1)} bpm and 95th percentile "
            f"{num(robust['abs_rate_difference_bpm_plausible_pairs']['95'], 1)} bpm; {pct(robust['share_within_10_bpm_plausible_pairs'])} lie within "
            f"±{target} bpm. Percentile limits of agreement (2.5th to 97.5th percentile, this run): {interval([limits['2.5'], limits['97.5']], 1)} bpm, "
            f"wider than the classical limits {interval([classical['loa_low'], classical['loa_high']], 1)} bpm. Gate G3 keeps its pre-specified rule and stays not met.")
    recordings = [r for r in read_csv(run, "per_recording.csv") if r["split"] == "test"]
    swaps = [r for r in read_csv(run, "swaps.csv") if r["split"] == "test"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 21)
    axes[0].hist([float(r["locked_share"]) for r in recordings], bins, alpha=HALF, label="same recording", density=True)
    axes[0].hist([float(r["maternal_locked_share"] or 0) for r in recordings], bins, alpha=HALF, label="mother's beats", density=True)
    axes[0].hist([float(r["locked_share"] or 0) for r in swaps], bins, alpha=HALF, label="another woman's fetal ECG", density=True)
    axes[0].set_xlabel("Share of beats that agree within 10 ms")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Trust statistic, held-out women")
    axes[0].legend()
    mean_rate = [(float(r["rate_ecg_bpm"]) + float(r["rate_doppler_bpm"])) / 2 for r in recordings]
    diff = [float(r["rate_doppler_bpm"]) - float(r["rate_ecg_bpm"]) for r in recordings]
    passing = np.array([r["trusted"] == "True" for r in recordings])
    axes[1].scatter(np.array(mean_rate)[passing], np.array(diff)[passing], s=18, label="passes the check")
    axes[1].scatter(np.array(mean_rate)[~passing], np.array(diff)[~passing], s=28, marker="x", color="tab:red", label="flagged")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Mean of the two heart rates (bpm)")
    axes[1].set_ylabel("Doppler minus ECG (bpm)")
    axes[1].set_title("Heart rate per recording, held-out women")
    axes[1].legend()
    picture = save(figure, "f3_trust_check", [run])
    return "\n\n".join(["## 3. ECG-Doppler trust check (NInFEA, real pairs)", *notes, text, gap_text, perturb_text, picture])


def section_wp2b() -> str:
    run = "wp2b_function_reference"
    metrics, _ = load_run(run)
    rows = []
    for name, v in metrics["values"].items():
        q = metrics["repeatability"][name]
        rows.append([name, f"{v['n_readable']}/{metrics['n_recordings']}", f"{num(v['median'], 2)} [{num(v['q25'], 2)}, {num(v['q75'], 2)}]",
                     num(q["split_half_abs_diff_median"], 2), f"{num(q['icc'], 2)} {interval(q['icc_ci'], 2)}", q["women_with_repeats"],
                     num(q["within_woman_sd"], 2)])
    text = table("t4_doppler_timing_measures",
                 ["Measure", "Readable", "Median [quartiles]", "First half vs second half, median abs. difference",
                  "ICC between recordings of a woman [95% CI]", "Women with repeats", "Within-woman SD"], rows, [run])
    flagged_rows = [[alpha, f["recordings_flagged"], f["recordings"], pct(f["share"]), interval(f["share_ci_by_woman"]),
                     f"{f['women_with_a_flagged_recording']}/{f['women']}"] for alpha, f in metrics["healthy_recordings_flagged"].items()]
    flagged_text = table("t4b_healthy_recordings_flagged", ["Alarm level", "Flagged", "Recordings", "Share", "95% CI (by woman)", "Women with a flagged recording"],
                         flagged_rows, [run])
    shift_rows = []
    for measure, entries in metrics["detectable_shift"].items():
        for key, value in entries.items():
            shift_rows.append([measure, key.replace("_", " "), value["smallest_shift_ms_for_50pct"] or "not within the range tried",
                               value["smallest_shift_ms_for_80pct"] or "not within the range tried"])
    shift_text = table("t4c_detectable_shift", ["Measure", "Direction and alarm level", "Shift flagged in half of the recordings (ms)",
                                                "Shift flagged in 80% (ms)"], shift_rows, [run])
    notes = [metrics["statement"],
             f"Usable recordings: {metrics['usable_recordings']} of {metrics['n_recordings']} ({metrics['usable_women']} women). "
             f"Smallest p-value possible with this many healthy recordings: {num(metrics['smallest_p_value_possible'], 3)}.",
             "The shift table is a simulation: a measured healthy value plus a stated shift of one interval. It shows what size of change the reference could flag, not what disease looks like."]
    figure, axis = plt.subplots(figsize=(8, 4))
    names = list(metrics["repeatability"])
    icc = [metrics["repeatability"][n]["icc"] for n in names]
    low = [metrics["repeatability"][n]["icc"] - metrics["repeatability"][n]["icc_ci"][0] for n in names]
    high = [metrics["repeatability"][n]["icc_ci"][1] - metrics["repeatability"][n]["icc"] for n in names]
    axis.errorbar(icc, range(len(names)), xerr=[low, high], fmt="o", capsize=3)
    axis.set_yticks(range(len(names)), names)
    axis.axvline(HALF, color="grey", linestyle="--")
    axis.set_xlabel("ICC between recordings of the same woman (95% CI)")
    axis.set_title("Repeatability of the Doppler timing measures")
    picture = save(figure, "f4_doppler_measure_repeatability", [run])
    return "\n\n".join(["## 4. Function expert: Doppler timing in healthy fetuses (NInFEA)", *notes, text, flagged_text, shift_text, picture])


def section_wp4() -> str:
    beats, intervals, foundation, cnn = "wp4a_nifeadb_beats", "wp4b_rhythm_intervals", "wp4c_foundation", "wp4d_cnn"
    m_beats, _ = load_run(beats)
    m_int, _ = load_run(intervals)
    m_found, _ = load_run(foundation)
    m_cnn, _ = load_run(cnn)
    rows, forest_rows = [], []

    def add(group: str, name: str, r: dict, p_text: str, run: str, role: str = "") -> None:
        extra = f"{num(r['sensitivity'], 2)} / {num(r['specificity'], 2)}" if "sensitivity" in r else "-"
        rows.append([group, name, r["n"], f"{num(r['auroc'])} {interval(r['auroc_ci'])}", p_text, extra, role, run])
        forest_rows.append((f"{group}: {name}", r["auroc"], r["auroc_ci"]))

    for name, r in m_int["recording_level"].items():
        add("interval features", name, r, f"{num(r['permutation_p'], 4)} (refit, {r['n_permutations']})", intervals, r.get("role", ""))
    for name, r in m_int["single_index"].items():
        add("single index, nothing fitted", name, r, f"{num(r['permutation_p'], 4)} ({r['n_permutations']})", intervals, r.get("role", r.get("note", "")))
    for name, r in m_found["arrhythmia_probe_nifeadb"].items():
        add("frozen network features", name.replace("__", " / "), r, f"{num(r['permutation_p'], 3)} (refit, {r['n_permutations']})", foundation, r["role"])
    add("network trained here", "small CNN", m_cnn["recording_level_leave_one_subject_out"],
        f"{num(m_cnn['recording_level_leave_one_subject_out']['mann_whitney_p_one_sided'], 3)} (Mann-Whitney)", cnn)
    text = table("t5_rhythm_expert", ["Group", "Model or index", "n", "AUROC [95% CI]", "p-value (method)", "Sensitivity / specificity at 0.5",
                                      "Role or note", "Run"], rows, [intervals, foundation, cnn])

    leak_rows = []
    for model in ("forest", "logistic"):
        v = m_int["leakage_comparison"][model]
        leak_rows.append([f"{model} on segment features", f"{num(v['segment_split_auroc'])} {interval(v['segment_split_ci'])}",
                          f"{num(v['subject_split_auroc'])} {interval(v['subject_split_ci'])}", num(v["gap"], 3), intervals])
    v = m_cnn["leakage_comparison_window_level"]
    leak_rows.append(["small CNN on 10-second windows", f"{num(v['segment_split_auroc'])} {interval(v['segment_split_ci'])}",
                      f"{num(v['subject_split_auroc'])} {interval(v['subject_split_ci'])}", num(v["gap"], 3), cnn])
    leak_text = table("t5b_leakage_comparison", ["Model", "Segments of a recording on both sides [95% CI]", "Recording held out [95% CI]",
                                                 "Inflation", "Run"], leak_rows, [intervals, cnn])

    probe_rows = [[key.replace("__", " / "), r["n"], f"{num(r['mae_bpm'], 2)} {interval(r['mae_ci'], 2)}", num(r["mae_of_mean_predictor_bpm"], 2),
                   num(r["r2"], 2), num(r["spearman"], 2)] for key, r in m_found["fetal_heart_rate_probe_cinc2013"].items()]
    probe_text = table("t5c_heart_rate_probe", ["Network / input", "Recordings", "Mean abs. error, bpm [95% CI]", "Error of predicting the mean",
                                                "R squared", "Spearman"], probe_rows, [foundation])
    check_rows = [[key.replace("__", " / "), num(r["auroc_all"]), num(r["auroc_without_500hz_recordings"]), r["n_without"],
                   num(r["mean_score_500hz_arrhythmia"], 2), num(r["mean_score_1khz_arrhythmia"], 2), num(r["mean_score_normal"], 2)]
                  for key, r in m_found["sampling_rate_check_added_after_first_run"].items()]
    check_text = table("t5d_sampling_rate_check", ["Network / input", "AUROC, all 26", "AUROC without the four 500 Hz recordings", "n",
                                                   "Mean score, 500 Hz arrhythmia", "Mean score, 1 kHz arrhythmia", "Mean score, normal"],
                       check_rows, [foundation])
    checkpoint = m_found["checkpoint"]
    gate = m_found["gate_G4"]
    usable = m_beats["by_label"]
    notes = [
        f"Gate G4 (every row has n, an interval and a run folder; the checkpoint matches the published file; negative-control rows present): **{state(gate['passed'])}**.",
        f"Checkpoint: {checkpoint['claimed_name']}, file {checkpoint['file']}, SHA-256 {checkpoint['sha256'][:16]}..., {checkpoint['n_parameters']:,} parameters, loaded with strict key matching.",
        f"Usable one-minute segments: {usable['arrhythmia']['usable_segments']}/{usable['arrhythmia']['n_segments']} in arrhythmia recordings, "
        f"{usable['normal']['usable_segments']}/{usable['normal']['n_segments']} in normal recordings; median extractor score "
        f"{num(usable['arrhythmia']['score_median'], 2)} against {num(usable['normal']['score_median'], 2)}.",
        "The sampling-rate table was added after the first run showed the maternal-lead control above chance; it re-summarises the same out-of-fold scores and is a check made after seeing a result.",
        "The small CNN was trained three times with different seeds; recording-level AUROC "
        + ", ".join(num(value) for value in m_cnn["recording_level_auroc_by_seed"].values())
        + f" (mean {num(np.mean(list(m_cnn['recording_level_auroc_by_seed'].values())))}). The first is the run shown in the table; the spread is training noise.",
    ]
    forest_rows = forest_rows[::-1]
    figure, axis = plt.subplots(figsize=(9, 9))
    position = np.arange(len(forest_rows))
    centre = np.array([r[1] for r in forest_rows])
    axis.errorbar(centre, position, xerr=[centre - np.array([r[2][0] for r in forest_rows]), np.array([r[2][1] for r in forest_rows]) - centre],
                  fmt="o", capsize=2)
    axis.axvline(HALF, color="grey", linestyle="--")
    axis.set_yticks(position, [r[0] for r in forest_rows], fontsize=7)
    axis.set_xlabel("AUROC with 95% CI: arrhythmia against normal rhythm, 26 subjects,\neach scored by a model that never saw it")
    axis.set_title("Rhythm expert and controls (NIFEADB)")
    picture = save(figure, "f5_rhythm_expert", [intervals, foundation, cnn])
    extra = []
    if (RUNS / CHECKS_RUN / "metrics.json").exists():
        checks = load_run(CHECKS_RUN)[0]
        rows_1k = read_csv(CHECKS_RUN, "rhythm_1khz_only.csv")
        khz_rows = []
        for r in rows_1k:
            if r["exact_from_saved_file"] == "True":
                khz_rows.append([r["description"], f"{num(r['auroc_all'])} {interval([r['ci_all_low'], r['ci_all_high']])}",
                                 f"{num(r['auroc_1khz'])} {interval([r['ci_1khz_low'], r['ci_1khz_high']])}", f"{r['n_1khz']} ({r['arrhythmia_1khz']} arrhythmia)"])
            else:
                khz_rows.append([r["description"], f"{num(r['published_auroc_all'])} (published)", "not recomputed: " + r["note"].split(": ")[-1], "-"])
        rates = checks["A_rhythm_1khz_only"]["sampling_rates_from_headers"]
        extra.append(
            "**Recordings sampled at 1 kHz only (added after seeing the results, run " + CHECKS_RUN + ").** Sampling rates read from the record headers: "
            + "; ".join(f"{fs}: {v['arrhythmia']} arrhythmia, {v['normal']} normal" for fs, v in rates.items())
            + ". The same out-of-fold scores are evaluated without the 500 Hz recordings, with a bootstrap interval over subjects; nothing is re-fitted. "
            "The saved per-record files hold scores rounded to 6 decimals; a score is recomputed only if its AUROC over all 26 recordings equals the published value exactly.")
        extra.append(table("t5e_rhythm_1khz_only", ["Score", "AUROC, all 26 [95% CI]", "AUROC, 1 kHz only [95% CI]", "Recordings (1 kHz)"], khz_rows, [CHECKS_RUN]))
        pairs = {(p["subset"], p["first"], p["second"]): p for p in checks["B_rhythm_paired_differences"]["rows"]}
        founder, twin = m_found["arrhythmia_probe_nifeadb"]["ecgfounder__fetal"], m_found["arrhythmia_probe_nifeadb"]["random_init__fetal_stretch2"]
        both = pairs[("all 26 recordings", "oof_ecgfounder__fetal", "oof_random_init__fetal_stretch2")]
        expert_1k = next(r for r in rows_1k if r["score"] == "oof_logistic_rhythm")
        extra.append(
            f"The network with random weights (stretched fetal input, {num(twin['auroc'])} {interval(twin['auroc_ci'])}) overlaps the interval of ECGFounder "
            f"({num(founder['auroc'])} {interval(founder['auroc_ci'])}); their paired difference is {num(both['difference'])} {interval([both['ci_low'], both['ci_high']])} "
            f"(run {CHECKS_RUN}, added after seeing the results), "
            + ("so the two cannot be ranked. " if not both["can_be_ranked"] else "so ECGFounder ranks above it. ")
            + f"On the 1 kHz recordings the rhythm expert fixed in the plan reaches {num(expert_1k['auroc_1khz'])} "
            f"{interval([expert_1k['ci_1khz_low'], expert_1k['ci_1khz_high']])}"
            + ("; the interval includes chance, so part of its AUROC on all 26 recordings may come from the sampling rate."
               if float(expert_1k["ci_1khz_low"]) <= HALF else "; the interval excludes chance."))
    return "\n\n".join(["## 5. Rhythm expert (NIFEADB, 12 arrhythmia and 14 normal)", *notes, text, leak_text, probe_text, check_text, *extra, picture])


PROBE_NAMES = {
    "fetalclip_scan": "FetalCLIP, de-identified scan (the structure expert)",
    "fetalclip_unaltered_image": "FetalCLIP, image as released",
    "random_weights_scan": "same network with random weights (control)",
    "fetalclip_overlay_only": "FetalCLIP, overlays only (shortcut audit)",
    "no_anatomy_numbers": "six numbers without anatomy (shortcut audit)",
}


def structure_runs() -> list[str]:
    """Structure runs on open data (DELTA). Runs on restricted data are reported in section 13 only."""
    return sorted(path.name for path in RUNS.glob("wp5_structure_*") if (path / "metrics.json").exists()
                  and not load_run(path.name)[1]["config"].get("restricted"))


def section_structure(run: str) -> str:
    metrics, _ = load_run(run)
    data, own, same, alarm = metrics["dataset"], metrics.get("own_partition") or {}, metrics.get("same_image_size") or {}, metrics["alarm"]
    name = data["name"]

    def cell(result) -> str:
        return f"{num(result['auroc'])} {interval(result['ci'])}"

    probe_rows = [[label, cell(metrics["probes"][key]), num(metrics["probes"][key].get("p_value"), 3),
                   cell(own[key]) if key in own else "-", cell(same["probes"][key]) if key in same.get("probes", {}) else "-"]
                  for key, label in PROBE_NAMES.items()]
    probe_text = table(f"t5_structure_probes_{name}",
                       ["Input to the probe", "AUROC, repeated case-level folds [95% CI]", "p-value (labels shuffled, probe refitted)",
                        "AUROC on the dataset's own test set [95% CI]", f"AUROC inside the most common image size, {same.get('size', '-')} [95% CI]"],
                       probe_rows, [run])

    def alarm_row(label: str, entry: dict) -> list:
        hit, false = entry["sensitivity"], entry["false_alarms"]
        return [label, entry["alpha"], f"{hit['caught']}/{hit['n']}", pct(hit["share"]), interval(hit["ci"], 2),
                f"{false['flagged']}/{false['n']}", pct(false["share"]), interval(false["ci"], 3)]

    alarm_rows = [alarm_row("repeated case-level folds, all cases", alarm)]
    if "fetalclip_scan" in own:
        alarm_rows.append(alarm_row("dataset's own partition: fitted on Train, calibrated on Val, judged on Test", own["fetalclip_scan"]["alarm"]))
    alarm_text = table(f"t5b_structure_alarm_{name}",
                       ["Evaluation", "Alarm level", "Abnormal cases caught", "Sensitivity", "95% CI", "Normal cases flagged", "False-alarm share", "95% CI"],
                       alarm_rows, [run])
    subtype_text = table(f"t5c_structure_subtypes_{name}", ["Defect subtype", "Cases", "Caught at the alarm level", "95% CI of the share"],
                         [[s["subtype"], s["cases"], s["caught"], interval(s["ci"], 2)] for s in alarm["by_subtype"]], [run])
    plane_text = table(f"t5d_structure_planes_{name}", ["Plane used alone", "Cases", "Abnormal", "AUROC [95% CI]"],
                       [[p["plane"], p["cases"], p["abnormal"], cell(p) if "auroc" in p else p.get("note", "-")] for p in metrics["by_plane"]], [run])
    size_text = table(f"t5e_structure_image_sizes_{name}", ["Image size", "Normal cases", "Abnormal cases", "Abnormal share"],
                      [[s["size"], s["normal"], s["abnormal"], pct(s["abnormal"] / (s["normal"] + s["abnormal"]))] for s in metrics["image_size_by_label"]], [run])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    keys = list(PROBE_NAMES)
    values = [metrics["probes"][key]["auroc"] for key in keys]
    lower = [metrics["probes"][key]["auroc"] - metrics["probes"][key]["ci"][0] for key in keys]
    upper = [metrics["probes"][key]["ci"][1] - metrics["probes"][key]["auroc"] for key in keys]
    colours = ["tab:blue", "tab:blue", "tab:gray", "tab:orange", "tab:orange"]
    axes[0].barh(np.arange(len(keys)), values, xerr=[lower, upper], color=colours, capsize=3)
    axes[0].axvline(HALF, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(np.arange(len(keys)), ["FetalCLIP\nde-identified scan", "FetalCLIP\nimage as released", "random weights\n(control)",
                                              "overlays only\n(shortcut audit)", "numbers without anatomy\n(shortcut audit)"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("AUROC, normal against abnormal (case-level folds, 95% CI)")
    axes[0].set_title(f"{name.upper()}: {data['abnormal_cases']} abnormal and {data['normal_cases']} normal cases")
    subtypes = alarm["by_subtype"]
    axes[1].bar(np.arange(len(subtypes)), [s["caught"] / s["cases"] for s in subtypes], color="tab:blue")
    for position, s in enumerate(subtypes):
        axes[1].text(position, s["caught"] / s["cases"], f"{s['caught']}/{s['cases']}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(np.arange(len(subtypes)), [s["subtype"] for s in subtypes], rotation=45)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel(f"Share caught at alarm level {alarm['alpha']}")
    axes[1].set_title("Cases caught, by defect subtype (few cases each)")
    picture = save(figure, f"f7_structure_expert_{name}", [run])

    controls_text = ""
    controls_run = f"wp5b_structure_controls_{name}"
    if (RUNS / controls_run / "metrics.json").exists():
        controls, _ = load_run(controls_run)
        diff = controls["A_paired_auroc_differences"]
        diff_rows = [[PROBE_NAMES[key], num(entry["auroc_control"]), f"{num(entry['difference'])} {interval(entry['ci'])}",
                      "yes" if entry["interval_excludes_zero"] else "no"] for key, entry in diff.items()]
        b, c, e = controls["B_overlay_only_inside_one_image_size"], controls["C_alarms_of_expert_and_overlay_probe"], controls["E_plane_without_the_heart"]
        controls_text = "\n\n".join([
            "**Checks made after seeing the results above** (run " + controls_run + "): the overlay-only probe scored close to the structure expert, so the gap was measured on the same cases.",
            table(f"t5f_structure_controls_{name}", ["Control", "AUROC of the control", "Structure expert minus control [95% CI, bootstrap over cases]", "Interval excludes zero"],
                  diff_rows, [controls_run]),
            f"Inside the {b['size']} images ({b['cases']} cases, {b['abnormal']} abnormal) the overlay-only probe reached {num(b['overlay_only']['auroc'])} "
            f"{interval(b['overlay_only']['ci'])} against {num(b['structure_expert_same_cases']['auroc'])} for the structure expert. "
            f"At the alarm level {c['alpha']}, the structure expert caught {c['caught_by_structure_expert']} of {c['abnormal_cases']} abnormal cases and the overlay-only probe "
            f"{c['caught_by_overlay_only_probe']}; {c['caught_by_both']} were caught by both and {c['caught_by_structure_expert_only']} by the structure expert alone. "
            f"The abdominal plane, which does not show the heart, reached {num(e['abdomen']['auroc'])} {interval(e['abdomen']['ci'])} on its own "
            f"(four-chamber plane {num(e['four_chamber']['auroc'])} {interval(e['four_chamber']['ci'])}).",
        ])

    record = metrics["models"]["published"]
    notes = [
        f"Dataset: {name.upper()} ({data['source']}). {data['images']} still images of {data['cases']} cases: {data['abnormal_cases']} abnormal, {data['normal_cases']} normal. "
        "All images of a case stay on one side of every split. Probe, folds and alarm level were fixed in the script before any image was processed.",
        f"Model: {record['claimed_name']} (SHA-256 {record['sha256'][:16]}..., {record['n_parameters']:,} parameters, licence {record['licence']}), frozen; "
        "control: the same network with random weights.",
        f"Shortcut audit: {metrics['shortcut_audit']['reading']}. Image size differs between the classes (table below), so the main probes are repeated inside the most common size.",
        "The subtype table has few cases per row; it is descriptive only.",
    ]
    parts = [f"### Labelled data: {name.upper()}", *notes, probe_text, alarm_text, plane_text, subtype_text, size_text]
    return "\n\n".join(part for part in [*parts, controls_text, picture] if part)


def section_wp5() -> str:
    labelled = structure_runs()
    controls = [path.name for path in RUNS.glob("wp5b_structure_controls_*") if (path / "metrics.json").exists()]
    if not labelled:
        return ("## 6. Structure expert (fetal echo)\n\n"
                "**No labelled echo data is on disk, so there is no result on heart defects.** See docs/data_cards/echo.md.")
    MANIFEST.append({"file": "RESULTS.md#structure", "runs": [*labelled, *controls]})
    lines = ["Gate G5 (the released reference model reproduced; pooled patient-level AUROC with an interval; shortcut audit reported): "
             "**partly met** - the AUROC with its interval and the shortcut audit are reported; no released reference model exists for the dataset used. "
             "**The shortcut audit finds label information outside the heart**: probes that see little or no cardiac anatomy separate the classes well above chance (tables below). "
             "The structure expert's AUROC is therefore an upper bound on what heart structure contributes, not a measure of it.",
             *[section_structure(name) for name in labelled]]
    return "\n\n".join(["## 6. Structure expert (fetal echo)", *lines])


def section_wp6b() -> str:
    run = "wp6b_fusion_structure"
    if not (RUNS / run / "metrics.json").exists():
        return ""
    metrics, _ = load_run(run)
    healthy, h7 = metrics["B_healthy_people_with_three_tests"], metrics["hypothesis_H7_structure"]
    rates, combined = healthy["false_alarm_rate_of_each_expert"], healthy["false_referral_rate_three_tests"]
    healthy_text = table("t6d_three_tests_healthy", ["How the three healthy scores are combined", "False-referral rate", "Ceiling from the alarm levels", "Within the ceiling"],
                         [[key.replace("_", " "), pct(value), pct(healthy["ceiling_sum_of_levels"], 0), healthy["ceiling_respected"][key]] for key, value in combined.items()], [run])
    comparison = read_csv(run, "rule_comparison_structure.csv")
    names = ["maximum", "noisy_or", "mean", "product_of_experts", "dempster"]
    rule_rows = [[r["target_false_referral_rate"], r["availability"], r["coupling"], r["n_virtual_abnormal"], r["n_real_scores"]]
                 + [f"{num(r[f'{n}_sensitivity'], 2)} {interval([r[f'{n}_ci_low'], r[f'{n}_ci_high']], 2)}" for n in names]
                 + [f"{num(r['max_minus_mean_sensitivity'], 2)} {interval([r['max_minus_mean_ci_low'], r['max_minus_mean_ci_high']], 2)}"]
                 for r in comparison]
    rule_text = table("t6e_rule_comparison_structural_defect",
                      ["Target false-referral rate", "Tests available", "Coupling", "Virtual patients with a defect", "Real scores used"]
                      + [f"{n.replace('_', ' ')}: sensitivity [95% CI]" for n in names] + ["maximum minus mean [95% CI]"], rule_rows, [run])
    notes = [
        "This part uses the structure expert's real scores. Nobody has all three tests, so scores of different people are combined; every such row is a virtual patient.",
        f"False-alarm rate of each expert on its own healthy cases: structure {pct(rates['structure'])}, function {pct(rates['function'])}, rhythm {pct(rates['rhythm'])}. "
        "The last row of the first table is the largest false-referral rate that any dependence between the tests could give.",
        f"Hypothesis H7 for a structural defect ({h7['statement']}): with mixed availability the mean rule had lower sensitivity than the OR rule in "
        f"{h7['rows_where_mean_rule_has_lower_sensitivity']} of {h7['rows_checked']} rows, and the interval of the difference excluded zero in {h7['rows_where_interval_excludes_zero']} rows. "
        f"With both tests always done: lower in {h7['both_tests_rows_where_mean_rule_has_lower_sensitivity']} of {h7['rows_checked']} rows, interval excluding zero in "
        f"{h7['both_tests_rows_where_interval_excludes_zero']}.",
    ]
    return "\n\n".join(["## 8. Decision rule with the structure expert's scores", *notes, healthy_text, rule_text])


def section_wp6() -> str:
    run = "wp6_fusion"
    metrics, _ = load_run(run)
    a, b, c = metrics["A_conformal_alarms"], metrics["B_real_paired_cohort"], metrics["C_rule_comparison"]
    rhythm, function = a["rhythm"], a["function"]
    alarm_rows = [
        ["rhythm (NIFEADB, leave-one-subject-out)", rhythm["alpha"], f"{rhythm['arrhythmia_caught']}/{rhythm['n_arrhythmia']}",
         interval(rhythm["sensitivity_ci"], 2), f"{rhythm['normal_flagged']}/{rhythm['n_normal']}", num(rhythm["smallest_possible_p"], 3)],
        ["function (NInFEA, leave-one-woman-out)", function["alpha"], "no abnormal case", "-",
         f"{function['flagged']}/{function['n_healthy_recordings']}", "-"],
        ["structure", "-", "no labelled data", "-", "-", "-"],
    ]
    alarm_runs = [run]
    if (RUNS / "wp6b_fusion_structure" / "metrics.json").exists():
        structure = load_run("wp6b_fusion_structure")[0]["A_alarms_of_three_experts"]["structure"]
        alarm_rows[-1] = [f"structure ({structure['dataset'].upper()}, case-level folds)", structure["alpha"],
                          f"{structure['caught']}/{structure['n_abnormal']}", interval(structure["sensitivity_ci"], 2),
                          f"{structure['normal_flagged']}/{structure['n_normal']}", "-"]
        alarm_runs.append("wp6b_fusion_structure")
    alarm_text = table("t6_conformal_alarms", ["Expert", "Alarm level", "Abnormal cases caught", "95% CI of sensitivity",
                                               "Healthy cases flagged", "Smallest possible p"], alarm_rows, alarm_runs)
    local = b["with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out"]
    cohort_rows = []
    for label, counts, false in (("rhythm expert as fitted on NIFEADB", b["actions_if_echo_were_done_and_normal"], b["false_referrals"]),
                                 ("rhythm score calibrated on healthy NInFEA recordings of other women", local["actions_if_echo_were_done_and_normal"], local["false_referrals"])):
        cohort_rows.append([label, counts["n"], counts["REFER"], counts["REPEAT"], counts["ROUTINE"], pct(false["share"]), interval(false["ci"], 3),
                            f"{false['by_expert']['rhythm']} / {false['by_expert']['function']}", num(false["ceiling_from_alarm_levels"], 2)])
    cohort_text = table("t6b_real_paired_cohort", ["Calibration of the rhythm expert", "Recordings", "REFER (false)", "REPEAT", "ROUTINE",
                                                   "False-referral share", "95% CI", "Alarms: rhythm / function", "Ceiling from the alarm levels"],
                        cohort_rows, [run])
    comparison = read_csv(run, "rule_comparison.csv")
    names = ["maximum", "noisy_or", "mean", "product_of_experts", "dempster"]
    rule_rows = [[r["target_false_referral_rate"], r["availability"], r["coupling"], r["n_virtual_arrhythmia"], r["n_real_scores"]]
                 + [f"{num(r[f'{n}_sensitivity'], 2)} {interval([r[f'{n}_ci_low'], r[f'{n}_ci_high']], 2)}" for n in names]
                 + [f"{num(r['max_minus_mean_sensitivity'], 2)} {interval([r['max_minus_mean_ci_low'], r['max_minus_mean_ci_high']], 2)}"]
                 for r in comparison]
    rule_text = table("t6c_rule_comparison_virtual_patients",
                      ["Target false-referral rate", "Tests available", "Coupling", "Virtual arrhythmia patients", "Real scores used"]
                      + [f"{n.replace('_', ' ')}: sensitivity [95% CI]" for n in names] + ["maximum minus mean [95% CI]"], rule_rows, [run])
    bounds, h7 = c["bounds"], metrics["hypothesis_H7"]
    as_is = b["actions_as_is_echo_missing"]
    notes = [
        f"Decision settings: alarm levels {metrics['decision_config']['alpha']}, priors {metrics['decision_config']['prior']} "
        f"(function and rhythm priors are assumptions), a missing test is requested when its prior is above {metrics['decision_config']['get_test_prior_limit']}.",
        f"Real paired cohort, as it is (no echo exists in NInFEA): REFER {as_is['REFER']}, REPEAT {as_is['REPEAT']}, GET_TEST {as_is['GET_TEST']}, ROUTINE {as_is['ROUTINE']} of {as_is['n']} recordings. "
        "The second table replaces the missing echo by an assumed normal result to show the ECG-Doppler part alone; that is a stated assumption.",
        b["caveat"], "With the rhythm score calibrated locally, the second row of the cohort table shows " + local["what_this_shows"],
        f"Bounds on real scores, all three couplings: false-referral rate {', '.join(f'{k} {pct(v)}' for k, v in bounds['false_referral_rate_by_coupling'].items())} "
        f"against the ceiling {pct(bounds['ceiling_sum_of_levels'], 0)} (respected: {bounds['ceiling_respected']}); "
        f"sensitivity never below the rhythm expert alone (respected: {bounds['sensitivity_floor_respected']}).",
        f"Hypothesis H7 ({h7['statement']}): the mean rule had lower sensitivity than the OR rule in {h7['rows_where_mean_rule_has_lower_sensitivity']} of {h7['rows_checked']} "
        f"mixed-availability rows, but the interval of the difference excluded zero in {h7['rows_where_interval_excludes_zero']} rows. "
        "With 12 arrhythmia and 14 normal rhythm scores the comparison is not conclusive. That one alarm cannot be cancelled is a property of the rule and is tested in tests/test_fusion.py.",
        "The rule comparison uses virtual patients made by pairing real held-out scores of different people. It is not a measurement of a fused system on patients.",
    ]
    cohort = read_csv(run, "paired_cohort_ninfea.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = {"REFER": "tab:red", "REPEAT": "tab:orange", "ROUTINE": "tab:green"}
    for position, key in enumerate(("action_if_echo_normal", "action_local_calibration")):
        bottom = 0
        for action, colour in colours.items():
            count = sum(r[key] == action for r in cohort)
            axes[0].bar(position, count, bottom=bottom, color=colour, label=action if position == 0 else None)
            if count:
                axes[0].text(position, bottom + count / 2, str(count), ha="center", va="center", color="white")
            bottom += count
    axes[0].set_xticks([0, 1], ["rhythm expert\nas fitted on NIFEADB", "rhythm score calibrated\non healthy NInFEA recordings"])
    axes[0].set_ylabel("Recordings (all fetuses healthy)")
    axes[0].set_title("Decision on real ECG-Doppler pairs (echo assumed normal)")
    axes[0].legend()
    mixed = [r for r in comparison if r["availability"] == "mixed" and r["target_false_referral_rate"] == comparison[0]["target_false_referral_rate"]]
    width = 1 / (len(names) + 1)
    for k, name in enumerate(names):
        axes[1].bar(np.arange(len(mixed)) + k * width, [float(r[f"{name}_sensitivity"]) for r in mixed], width, label=name.replace("_", " "))
    axes[1].set_xticks(np.arange(len(mixed)) + 2 * width, [r["coupling"] for r in mixed])
    axes[1].set_ylabel("Arrhythmia patients referred (virtual patients)")
    axes[1].set_title(f"Rules at a false-referral rate of {mixed[0]['target_false_referral_rate']}, half without Doppler")
    axes[1].legend(fontsize=7, ncol=len(names), loc="upper center", bbox_to_anchor=(HALF, -THIRD * HALF))
    picture = save(figure, "f6_fusion_decision", [run])
    return "\n\n".join(["## 7. Fusion and decision", *notes, alarm_text, cohort_text, rule_text, picture])


def pct_interval(pair, digits: int = 1) -> str:
    return f"[{pct(float(pair[0]), digits)}, {pct(float(pair[1]), digits)}]"


def levels_text(values: dict) -> str:
    return ", ".join(f"{name} {value}" for name, value in values.items())


def yes(value) -> str:
    return "yes" if str(value) == "True" else "no"


def section_guarantees() -> str:
    if not (RUNS / BOUNDS_RUN / "metrics.json").exists():
        return ""
    metrics, _ = load_run(BOUNDS_RUN)
    gate = metrics["gate_G7"]
    range_rows = []
    for r in read_csv(BOUNDS_RUN, "dependence_ranges.csv"):
        measured = "-" if r["measured_joint_rate"] == "" else f"{pct(float(r['measured_joint_rate']))} {pct_interval(json.loads(r['measured_joint_ci']))}"
        range_rows.append([r["combination"], r["people"], r["experts"], pct(float(r["ceiling"]), 0),
                           pct_interval([r["lowest_rate_any_dependence"], r["highest_rate_any_dependence"]]), pct(float(r["rate_if_independent"])),
                           pct_interval([r["lowest_with_uncertainty"], r["highest_with_uncertainty"]]), measured,
                           f"{yes(r['inside_ceiling_at_measured_rates'])} / {yes(r['inside_ceiling_with_uncertainty'])}"])
    range_text = table("t9_false_referral_range_any_dependence",
                       ["Combination", "People", "Alarms on healthy cases (each expert)", "Ceiling (sum of levels)", "Range for any dependence",
                        "If independent", "Range with uncertainty", "Measured on the same fetuses [95% CI]", "Inside the ceiling: measured rates / with uncertainty"],
                       range_rows, [BOUNDS_RUN])
    calibration_rows = [[c["expert"], c["calibration_set"], c["n_healthy"], c["alarm_level"], num(c["smallest_possible_p"], 3),
                         pct_interval([c["range_low"], c["range_high"]])] for c in metrics["realised_rate_with_a_fixed_calibration_set"]]
    calibration_text = table("t9b_realised_rate_fixed_calibration",
                             ["Expert", "Healthy calibration cases", "n", "Alarm level", "Smallest possible p", "Realised false-alarm rate (90% range)"],
                             calibration_rows, [BOUNDS_RUN])
    floor_rows = [[f["finding"], f["expert"], f"{f['caught']}/{f['n']}" if f["n"] != "" else "-",
                   f"{pct(f['sensitivity_floor'])} {pct_interval([f['ci_low'], f['ci_high']])}" if f["n"] != "" else "cannot be measured",
                   f["dataset"], f["note"]] for f in metrics["sensitivity_floor"]]
    floor_text = table("t9c_sensitivity_floor", ["Finding", "Expert", "Caught", "Detected by its own expert at its own level [95% CI]", "Dataset", "Note"],
                       floor_rows, [BOUNDS_RUN])
    grid = read_csv(BOUNDS_RUN, "precision_grid.csv")
    findings = list(dict.fromkeys(r["finding"] for r in grid))
    scenarios = (("precision_alarms_tied", "alarms tied"), ("precision_independent", "independent"),
                 ("precision_alarms_never_together", "alarms never together"), ("precision_pessimistic", "pessimistic"))
    precision_rows = []
    for prevalence in dict.fromkeys(r["prevalence"] for r in grid):
        cells = [pct(float(prevalence))]
        for finding in findings:
            row = next(r for r in grid if r["finding"] == finding and r["prevalence"] == prevalence)
            cells += [pct(float(row[key])) for key, _ in scenarios]
        precision_rows.append(cells)
    precision_text = table("t9d_precision_by_prevalence", ["Prevalence"] + [f"{finding}: {label}" for finding in findings for _, label in scenarios],
                           precision_rows, [BOUNDS_RUN])
    priors = metrics["priors"]
    notes = [
        "Added after seeing the results (runs " + BOUNDS_RUN + " and " + TRADEOFF_RUN + "); nothing is fitted. Two properties of the any-alarm rule hold "
        "without any paired patient, for any dependence between the tests, and property tests check that the code has them (tests/test_guarantees.py, "
        "tests/test_fusion.py). Both are standard, not contributions of this project (docs/positioning.md): (1) the share of healthy fetuses referred is at most "
        "the sum of the alarm levels of the experts used - the union (Bonferroni) bound; (2) a fetus whose finding makes its expert alarm is always referred. "
        "Property (2) only restates each expert's own sensitivity: combining tests widens which conditions are checked and adds false referrals, but it never "
        "makes any one condition easier to detect. The range between the largest single rate and the sum (the Frechet bounds) is equally standard.",
        f"Gate G7, defined {gate['defined']}: {gate['rule']}. At the measured rates **{state(gate['passed_at_measured_rates'])}**, range "
        f"{pct_interval(gate['range_at_measured_rates'])} against a ceiling of {pct(gate['ceiling'], 0)}; with uncertainty **{state(gate['passed_with_uncertainty'])}**, "
        f"range {pct_interval(gate['range_with_uncertainty'])} ({gate['uncertainty']}). Upper limit of each rate: "
        + ", ".join(f"{name} {pct(value)}" for name, value in gate["upper_limits_each"].items()) + ".",
        "Rows that combine different groups of people bound a fetus that nobody has examined with all three tests; only the ECG-Doppler rows are measured on "
        "the same fetuses. The range holds only if each expert's rate holds where it is used: the rhythm expert calibrated on another device referred far more "
        "healthy fetuses than its level (fourth row).",
        f"Not measured: {metrics['not_measured']}.",
        f"Priors used by the decision rule: {levels_text(priors['values'])}; {priors['status']}. Precision of a referral is therefore shown over a grid of prevalences. "
        "It uses the expert's own sensitivity (a floor for the rule) and the healthy false-referral rate of the first row; "
        "'pessimistic' combines the lower 95% limit of the sensitivity with the upper end of the range with uncertainty.",
    ]
    dial = read_csv(BOUNDS_RUN, "dependence_dial.csv")
    rho = np.array([float(r["rho"]) for r in dial])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(rho, [float(r["rate_at_measured_rates"]) for r in dial], marker="o", markersize=3, color="tab:blue", label="dial, measured rates")
    axis.plot(rho, [float(r["rate_at_upper_limits"]) for r in dial], linestyle="--", color="tab:purple", label="dial, upper limit of each rate")
    low, high = gate["range_at_measured_rates"]
    axis.axhline(low, color="tab:green", linewidth=1, label=f"lowest rate, any dependence ({pct(low)})")
    axis.axhline(high, color="tab:orange", linewidth=1, label=f"highest rate, any dependence ({pct(high)})")
    axis.axhline(gate["ceiling"], color="black", linestyle=":", label=f"design ceiling ({pct(gate['ceiling'], 0)})")
    axis.axhline(gate["range_with_uncertainty"][1], color="tab:red", linestyle="-.", linewidth=1,
                 label=f"highest rate with uncertainty ({pct(gate['range_with_uncertainty'][1])})")
    axis.set_xlabel("Correlation shared by every pair of experts (Gaussian copula; 0 = independent, 1 = tied)")
    axis.set_ylabel("Share of healthy fetuses referred")
    axis.set_title("False referrals of three experts under different dependence")
    axis.legend(fontsize=7)
    picture = save(figure, "f8_dependence_dial", [BOUNDS_RUN])
    parts = ["## 9. What the any-alarm rule guarantees without paired data (standard bounds)", *notes, range_text, calibration_text, floor_text,
             precision_text, picture]

    if (RUNS / TRADEOFF_RUN / "metrics.json").exists():
        trade, provenance = load_run(TRADEOFF_RUN)
        levels = provenance["config"]["levels"]
        curves = read_csv(TRADEOFF_RUN, "expert_curves.csv")
        level_rows = []
        for r in curves:
            if not any(np.isclose(float(r["alarm_level"]), level) for level in levels[r["expert"]]):
                continue
            level_rows.append([r["expert"], num(r["alarm_level"], 3), f"{r['healthy_flagged']}/{r['healthy_n']}",
                               f"{pct(float(r['healthy_rate']))} {pct_interval([r['healthy_ci_low'], r['healthy_ci_high']])}",
                               f"{r['finding_caught']}/{r['finding_n']}" if r["finding_n"] else "-",
                               f"{pct(float(r['sensitivity']))} {pct_interval([r['sensitivity_ci_low'], r['sensitivity_ci_high']])}" if r["finding_n"] else "no abnormal cases"])
        level_text = table("t9e_alarm_levels", ["Expert", "Alarm level", "Healthy flagged", "Healthy alarm rate [95% CI]", "Abnormal caught", "Sensitivity [95% CI]"],
                           level_rows, [TRADEOFF_RUN])
        design, smallest = trade["design_row"], trade["rhythm_smallest_level"]
        parts += [
            "**Alarm levels against sensitivity (run " + TRADEOFF_RUN + ", added after seeing the results).** " + trade["caution"].capitalize() + ". "
            f"The full grid of {trade['n_designs']} combinations of levels is in runs/{TRADEOFF_RUN}/designs.csv. At the design levels "
            f"({levels_text(trade['design_levels'])}) the ceiling is {pct(design['ceiling'], 0)}, the healthy range for any dependence "
            f"{pct_interval([design['healthy_rate_lowest_any_dependence'], design['healthy_rate_highest_any_dependence']])}, and the sensitivity "
            f"{pct(design['structural_defect_sensitivity'])} for a structural defect and {pct(design['arrhythmia_sensitivity'])} for an arrhythmia.",
            f"Rhythm expert: with {smallest['n_healthy_calibration']} healthy calibration recordings its p-values are multiples of "
            f"1/{smallest['n_healthy_calibration'] + 1}, so it cannot alarm below {num(smallest['smallest_p_for_a_new_case'], 3)}, and the design level "
            f"{trade['design_levels']['rhythm']} acts as {num(smallest['effective_level_of_the_design'], 3)}: the ceiling the design actually guarantees is "
            f"{pct(smallest['effective_ceiling_of_the_design'])}, not {pct(design['ceiling'], 0)}. In the leave-one-out evaluation each healthy recording is "
            f"ranked among the other {smallest['n_healthy_calibration'] - 1}, so at {num(smallest['smallest_p_for_a_new_case'], 3)} its measured healthy alarm "
            "rate is 0 by construction (table below).",
            level_text,
        ]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for expert, colour in (("structure", "tab:blue"), ("rhythm", "tab:orange"), ("function", "tab:green")):
            rows = [r for r in curves if r["expert"] == expert]
            x = [float(r["alarm_level"]) for r in rows]
            if rows[0]["finding_n"]:
                axes[0].plot(x, [float(r["sensitivity"]) for r in rows], color=colour, label=f"{expert}: sensitivity")
            axes[0].plot(x, [float(r["healthy_rate"]) for r in rows], color=colour, linestyle="--", label=f"{expert}: healthy alarm rate")
        axes[0].axvline(smallest["smallest_p_for_a_new_case"], color="grey", linestyle=":", label="smallest rhythm level that can alarm")
        axes[0].set_xlabel("Alarm level of the expert")
        axes[0].set_ylabel("Share")
        axes[0].set_title("What each alarm level buys")
        axes[0].legend(fontsize=7)
        designs = read_csv(TRADEOFF_RUN, "designs.csv")
        ceiling = [float(r["ceiling"]) for r in designs]
        axes[1].scatter(ceiling, [float(r["structural_defect_sensitivity"]) for r in designs], s=12, label="structural defect (structure expert)")
        axes[1].scatter(ceiling, [float(r["arrhythmia_sensitivity"]) for r in designs], s=12, marker="^", label="arrhythmia (rhythm expert)")
        axes[1].scatter([design["ceiling"]] * 2, [design["structural_defect_sensitivity"], design["arrhythmia_sensitivity"]], s=90, marker="*",
                        color="black", label="design levels")
        axes[1].set_xlabel("False-referral ceiling (sum of the three alarm levels)")
        axes[1].set_ylabel("Sensitivity (floor for the rule)")
        axes[1].set_title("Ceiling against sensitivity, every combination of levels")
        axes[1].legend(fontsize=7)
        parts.append(save(figure, "f9_ceiling_against_sensitivity", [TRADEOFF_RUN]))
    return "\n\n".join(parts)


def section_checks() -> str:
    if not (RUNS / CHECKS_RUN / "metrics.json").exists():
        return ""
    metrics, _ = load_run(CHECKS_RUN)
    subgroups = metrics["D_subgroups"]
    size_rows = [[r["image_size"], f"{r['normal_flagged']}/{r['normal']}",
                  pct_interval([r["fa_ci_low"], r["fa_ci_high"]]) if r["normal"] else "-",
                  f"{r['abnormal_caught']}/{r['abnormal']}", pct_interval([r["se_ci_low"], r["se_ci_high"]]) if r["abnormal"] else "-"]
                 for r in subgroups["structure_by_image_size"]]
    size_text = table("t10_structure_by_image_size", ["Image size", "Normal cases flagged", "95% CI", "Defects caught", "95% CI"], size_rows, [CHECKS_RUN])
    week_rows = [[r["gestation_weeks"], r["recordings"], r["women"], r["flagged"], pct_interval([r["ci_low"], r["ci_high"]])]
                 for r in subgroups["function_by_gestational_age"]]
    week_text = table("t10b_function_by_gestational_age", ["Gestational age (weeks+days)", "Recordings", "Women", "Flagged", "95% CI of the share"],
                      week_rows, [CHECKS_RUN])
    pair_rows = []
    for p in metrics["B_rhythm_paired_differences"]["rows"]:
        if "difference" in p:
            pair_rows.append([p["subset"], f"{p['first']} minus {p['second']}", p["n"], f"{num(p['difference'])} {interval([p['ci_low'], p['ci_high']])}",
                              "yes" if p["can_be_ranked"] else "no"])
        else:
            pair_rows.append([p["subset"], f"{p['first']} minus {p['second']}", p["n"], p["note"], "-"])
    pair_text = table("t10c_rhythm_paired_differences", ["Recordings", "Scores compared", "n", "AUROC difference [95% CI, paired bootstrap]", "Can be ranked"],
                      pair_rows, [CHECKS_RUN])
    correlation, missed = subgroups["function_score_and_gestational_age"], metrics["E_structure_missed"]
    notes = [
        "Added after seeing the results (run " + CHECKS_RUN + "). Every number re-summarises saved per-case outputs; nothing is re-fitted, and the p-values "
        "and intervals are not corrected for the number of checks.",
        f"Structure expert by image size, at its alarm level. {subgroups['not_available'].capitalize()}; image size is the only acquisition variable that can be checked.",
        f"Function expert by gestational age: Spearman correlation of its score with gestational age {num(correlation['spearman_score_vs_gestation_days'], 2)} "
        f"(p = {num(correlation['p_value'], 2)}, {correlation['n']} recordings; {correlation['note']}).",
        "Rhythm scores compared on the same subjects. 'slow_hr_bpm_median' is heart rate alone with lower values counted as arrhythmia, the direction seen "
        "in the data (section 5); choosing it after seeing the data favours this baseline.",
        f"Structural defects missed at the alarm level: {missed['n_missed']} of {missed['n_abnormal']}; their case identifiers (of the public dataset) are in "
        f"runs/{CHECKS_RUN}/structure_missed_cases.csv, and section 6 gives the share caught by defect type.",
    ]
    return "\n\n".join(["## 10. Checks made after seeing the results", *notes, size_text, week_text, pair_text])


def external_threat() -> str:
    if not have("wp11_external_echo_heartbeat_2T"):
        return ("11. **No external validation.** No expert has been tested on a dataset other than the one it was built on; the script for a "
                "second echo dataset is ready (scripts/run_wp11_external_echo.py, docs/preregistration_external_echo.md) but has no data.")
    t = load_run("wp11_external_echo_heartbeat_2T")[0]["transported_calibration"]
    return (f"11. **The structure expert failed its external test.** Applied unchanged to Heartbeat (another country, preregistered), it reached "
            f"AUROC {num(t['auroc'])} {interval(t['auroc_ci'])} and flagged {pct(t['false_alarms']['share'])} of normal fetuses at its level "
            "(section 13). The rhythm and function experts have still not been tested on a second dataset.")


def section_threats() -> str:
    needed = [BOUNDS_RUN, TRADEOFF_RUN, CHECKS_RUN, "wp6_fusion", "wp3_trust_check"]
    labelled = structure_runs()
    if not labelled or not all((RUNS / run / "metrics.json").exists() for run in needed):
        return ""
    structure, _ = load_run(labelled[0])
    bounds, trade, checks = load_run(BOUNDS_RUN)[0], load_run(TRADEOFF_RUN)[0], load_run(CHECKS_RUN)[0]
    fusion, trust = load_run("wp6_fusion")[0], load_run("wp3_trust_check")[0]
    probes = structure["probes"]
    controls = [probes[name]["auroc"] for name in ("no_anatomy_numbers", "fetalclip_overlay_only")]
    sizes = [r for r in checks["D_subgroups"]["structure_by_image_size"] if r["abnormal"] >= 10]
    expert_1k = next(r for r in read_csv(CHECKS_RUN, "rhythm_1khz_only.csv") if r["score"] == "oof_logistic_rhythm")
    cohort = fusion["B_real_paired_cohort"]
    local = cohort["with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out"]["false_referrals"]
    rhythm_range = next(c for c in bounds["realised_rate_with_a_fixed_calibration_set"] if c["expert"].startswith("rhythm, calibrated on NIFEADB"))
    gate, g3 = bounds["gate_G7"], trust["gate_G3"]
    beat = trust["by_split"]["test"]["beat_level_ecg_quality_passed_recordings"]["rate_diff_percentiles_plausible_bpm"]
    items = [
        "1. **Checks made after seeing the results.** Sections 9 and 10, gate G7 and the added rows in sections 3 and 5 were defined after the results were "
        "known. They re-summarise saved outputs; their intervals are not corrected for the number of checks.",
        f"2. **One dataset per expert.** Structure: {structure['dataset']['name'].upper()} ({structure['dataset']['cases']} cases, one centre); rhythm: NIFEADB "
        f"(26 recordings); function: NInFEA (healthy only). No result has been repeated on a second dataset.",
        f"3. **Acquisition can carry the label.** In DELTA, inputs with little or no cardiac anatomy reached AUROC {num(min(controls))} to {num(max(controls))}, and the "
        "structure expert caught " + "; ".join(f"{r['abnormal_caught']} of {r['abnormal']} defects in {r['image_size']} images" for r in sizes)
        + f". In NIFEADB every 500 Hz recording is an arrhythmia; on the 1 kHz recordings the rhythm expert reaches {num(expert_1k['auroc_1khz'])} "
        f"{interval([expert_1k['ci_1khz_low'], expert_1k['ci_1khz_high']])}.",
        f"4. **Calibration does not travel.** Calibrated on NIFEADB, the rhythm expert referred {pct(cohort['false_referrals']['share'])} of the healthy NInFEA "
        f"fetuses; calibrated on NInFEA, {pct(local['share'])}. The false-referral guarantee holds only for experts calibrated on healthy cases from the setting of use.",
        f"5. **Small samples.** With {rhythm_range['n_healthy']} healthy calibration recordings the rhythm expert's realised false-alarm rate lies between "
        f"{pct(rhythm_range['range_low'])} and {pct(rhythm_range['range_high'])} (90% range), and the false-referral range for any dependence widens to "
        f"{pct_interval(gate['range_with_uncertainty'])} with uncertainty (gate G7).",
        "6. **No abnormal Doppler recordings.** The function expert's sensitivity is unknown; the shifts it could flag come from simulation (section 4).",
        f"7. **No paired patients.** {bounds['not_measured'].capitalize()} are not measured. Only the ECG-Doppler pair was measured on the same fetuses (60 healthy recordings).",
        f"8. **ECG-Doppler agreement.** Gate G3 is not met: classical limits {interval(g3['beat_level_limits_bpm'], 1)} bpm, percentile limits "
        f"{interval([beat['2.5'], beat['97.5']], 1)} bpm.",
        f"9. **Assumed priors.** {bounds['priors']['status'].capitalize()}; precision is shown over a grid of prevalences (section 9).",
        f"10. **Discrete p-values.** With few calibration cases an alarm level acts as the next lower attainable p-value; for the rhythm expert the level "
        f"{trade['design_levels']['rhythm']} acts as {num(trade['rhythm_smallest_level']['effective_level_of_the_design'], 3)}.",
        external_threat(),
        "12. **Standard methods.** The union bound, the Frechet range, conformal p-values, group-conditional calibration, Holm and Simes, and choosing "
        "levels to maximise true positives under a false-positive cap are all established; docs/positioning.md and docs/related_work.md list what is "
        "and is not new.",
    ]
    if have(RATE_RUN):
        rate = load_run(RATE_RUN)[0]
        c = rate["comparison_with_original"]["logistic_rhythm"]
        items.append(f"13. **Recording setting in NIFEADB.** With every recording brought to 500 Hz the rhythm expert reaches {num(c['auroc_500hz'])} "
                     f"{interval(c['auroc_500hz_ci'])}, so the sampling rate itself was not its cue; but the mother's chest lead, which holds no fetal signal, "
                     f"still scores {num(rate['recording_level']['ecgfounder__maternal_chest']['auroc'])}: the four recordings sampled at 500 Hz, all "
                     "arrhythmia, differ in some other way that resampling cannot remove (section 12.1).")
    if have(TRUST_BASELINE_RUN):
        hr = load_run(TRUST_BASELINE_RUN)[0]["by_women"]["test"]["maternal_substitution"]["heart_rate_only"]
        items.append(f"14. **Simple baselines for the trust check.** Heart rate alone separates the mother's beats with AUROC {num(hr['auroc'])} "
                     f"{interval(hr['ci'])}, and a cross-correlation of the beat trains beats the locked share on swapped pairs (section 12.3).")
    return "\n\n".join(["## 11. Threats to validity", "\n".join(items)])


RATE_RUN, SIZE_RUN, TRUST_BASELINE_RUN = "wp4e_rhythm_500hz", "wp5c_structure_by_size", "wp3b_trust_baselines"
ALLOCATION_RUN, THEORY_RUN, TRANSFER_RUN = "wp10_allocation", "wp12_theory_simulation", "wp13_calibration_transfer"
CONFIRMATORY = "confirmatory: split, model and level fixed before the held-out data"
EXPLORATORY = "exploratory: made after seeing the results; interval not corrected for the number of checks"


def have(*runs: str) -> bool:
    return all((RUNS / run / "metrics.json").exists() for run in runs)


def claim_ledger() -> str:
    """One row per claim: how it was tested, on how many cases, with what result, and what would overturn it."""
    rows, runs = [], []

    def add(claim, kind, n, estimate, dataset, run, falsifier):
        rows.append([claim, kind, n, estimate, dataset, run, falsifier])
        runs.append(run)

    if have("wp1_cinc2013"):
        m = load_run("wp1_cinc2013")[0]
        test = m["by_config"]["full"]["test"]
        add("Fetal beats are found without following the mother", CONFIRMATORY, f"{test['n_recordings']} test recordings",
            f"median F1 {num(test['f1_median'])} {interval(test['f1_median_ci'])}", "CinC 2013 set A", "wp1_cinc2013",
            f"failing the gate on new recordings ({m['gate_G1']['rule']})")
    if have("wp2_doppler_cycles"):
        m = load_run("wp2_doppler_cycles")[0]
        test = m["by_config"]["full"]["test"]
        add("Cardiac cycles are found on Doppler strips", CONFIRMATORY, f"{test['n_strips']} test strips, {test['n_marks']} marks",
            f"mean F1 {num(test['f1_mean'])} {interval(test['f1_mean_ci'])}", "annotated NInFEA strips", "wp2_doppler_cycles",
            f"failing the gate on new strips ({m['gate_G2']['rule']})")
    if have("wp3_trust_check"):
        m = load_run("wp3_trust_check")[0]
        test, gate = m["by_split"]["test"], m["gate_G3"]
        falsifier = "a simpler score separating them as well"
        if have(TRUST_BASELINE_RUN):
            hr = load_run(TRUST_BASELINE_RUN)[0]["by_women"]["test"]["maternal_substitution"]["heart_rate_only"]
            falsifier += f" - heart rate alone reaches {num(hr['auroc'])} {interval(hr['ci'])} (section 12.3)"
        add("The trust check tells a real ECG-Doppler pair from the mother's beats", CONFIRMATORY,
            f"{test['auroc_vs_maternal']['n_pos']} recordings of {test['n_women']} test women",
            f"AUROC {num(test['auroc_vs_maternal']['auroc'])} {interval(test['auroc_vs_maternal']['ci'])}", "NInFEA", "wp3_trust_check", falsifier)
        add("ECG and Doppler beat-to-beat rates agree within the target", CONFIRMATORY, f"{test['n_women']} test women",
            f"limits {interval(gate['beat_level_limits_bpm'], 1)} bpm: **{state(gate['beat_level_passed'])}**", "NInFEA", "wp3_trust_check",
            "already falsified: gate G3 is not met")
    if have("wp4b_rhythm_intervals"):
        r = load_run("wp4b_rhythm_intervals")[0]["recording_level"]["logistic_rhythm"]
        falsifier = "AUROC near 0.5 on recordings of both classes from one recording setting"
        if have(RATE_RUN):
            w = load_run(RATE_RUN)[0]["without_native_500hz_recordings"]["logistic_rhythm"]
            falsifier += f" (the {w['n']} recordings from one setting give {num(w['auroc'])} {interval(w['auroc_ci'])}, section 12.1)"
        add("The rhythm expert separates arrhythmia from normal rhythm (feasibility on 26 recordings)", CONFIRMATORY,
            f"{r['n']} recordings ({r['n_positive']} arrhythmia)", f"AUROC {num(r['auroc'])} {interval(r['auroc_ci'])}", "NIFEADB",
            "wp4b_rhythm_intervals", falsifier)
    labelled = structure_runs()
    if labelled:
        m = load_run(labelled[0])[0]
        probe = m["probes"]["fetalclip_scan"]
        controls = [m["probes"][name]["auroc"] for name in ("no_anatomy_numbers", "fetalclip_overlay_only")]
        falsifier = (f"inputs without heart anatomy scoring as high (they reach {num(min(controls))} to {num(max(controls))}); "
                     "AUROC near 0.5 inside one image-size group or on a second dataset")
        if have(SIZE_RUN):
            flag = load_run(SIZE_RUN)[0]["flag_642x506"]
            falsifier += f" (642x506: {num(flag['auroc'])} {interval(flag['auroc_ci'])}, section 12.2)"
        if have("wp11_external_echo_heartbeat_2T"):
            external = load_run("wp11_external_echo_heartbeat_2T")[0]["transported_calibration"]
            falsifier += f"; on Heartbeat it reached {num(external['auroc'])} {interval(external['auroc_ci'])} (section 13.2)"
        else:
            falsifier += "; no second dataset yet"
        add("The structure expert reads heart structure (upper bound only)", CONFIRMATORY,
            f"{m['dataset']['cases']} cases ({m['dataset']['abnormal_cases']} with a defect)", f"AUROC {num(probe['auroc'])} {interval(probe['ci'])}",
            "DELTA (one centre)", labelled[0], falsifier)
    if have("wp2b_function_reference"):
        f = load_run("wp2b_function_reference")[0]["healthy_recordings_flagged"]
        design_level = load_run(TRADEOFF_RUN)[0]["design_levels"]["function"] if have(TRADEOFF_RUN) else min(map(float, f))
        level = next(k for k in f if np.isclose(float(k), design_level))
        row = f[level]
        add("The function expert flags healthy fetuses at no more than its level", CONFIRMATORY, f"{row['recordings']} healthy recordings",
            f"{row['recordings_flagged']} flagged at {level} {pct_interval(row['share_ci_by_woman'])} (by woman)", "NInFEA (healthy only)",
            "wp2b_function_reference", "more flagged than the level on new healthy recordings; its sensitivity is unknown")
    if have("wp6_fusion"):
        cohort = load_run("wp6_fusion")[0]["B_real_paired_cohort"]
        local = cohort["with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out"]["false_referrals"]
        transported = cohort["false_referrals"]
        add("False referrals stay under the ceiling when each expert is calibrated where it is used", CONFIRMATORY,
            f"{local['n']} real ECG-Doppler pairs", f"{pct(local['share'])} {pct_interval(local['ci'])} (ceiling {pct(local['ceiling_from_alarm_levels'], 0)})",
            "NInFEA", "wp6_fusion", "healthy pairs from a new setting referred above the ceiling after local calibration")
        add("Calibration carried over from another device does not hold", CONFIRMATORY, f"{transported['n']} real ECG-Doppler pairs",
            f"{pct(transported['share'])} {pct_interval(transported['ci'])} referred", "NIFEADB to NInFEA", "wp6_fusion",
            "carried-over calibration staying at its level on other devices")
    if have("wp11_external_echo_heartbeat_2T"):
        m = load_run("wp11_external_echo_heartbeat_2T")[0]
        t = m["transported_calibration"]
        v = external_verdict("wp11_external_echo_heartbeat_2T")
        kind = "confirmatory: preregistered on 20 Oct 2026, data seen only afterwards (clarifications of 27 Oct)"
        add("The structure expert reads heart structure in another country's data", kind,
            f"{m['cases']} patients ({t['n_abnormal']} with CHD)", f"AUROC {num(t['auroc'])} {interval(t['auroc_ci'])}: "
            + ("supports it" if v["reads"] else "does not support it"), "Heartbeat, 2nd trimester (Colombia)", "wp11_external_echo_heartbeat_2T",
            "already falsified for the frozen DELTA expert (section 13.2)" if not v["reads"] else "AUROC near 0.5 on a further dataset")
        add("The structure expert's DELTA alarm level holds in another country's data", kind, f"{t['false_alarms']['n']} normal patients",
            f"{t['false_alarms']['k']} flagged ({pct(t['false_alarms']['share'])}) {pct_interval(t['false_alarms']['ci'])} at level {v['level']}",
            "Heartbeat, 2nd trimester", "wp11_external_echo_heartbeat_2T",
            "already falsified: the interval excludes the level" if not v["calibration_holds"] else "an interval excluding the level")
    if have("wp17_structure_calibration"):
        d = load_run("wp17_structure_calibration")[0]["by_dataset"]["heartbeat_2T"]
        hold = d["smallest_that_holds"]["local"]
        if isinstance(hold, dict):
            add("Local calibration repairs the echo expert's false alarms but not its detection", EXPLORATORY,
                f"{d['n_healthy']} healthy and {d['n_chd']} CHD patients",
                f"false alarms within the level from {hold['healthy']} local healthy patients; {pct(hold['detection_mean'])} of CHD caught then",
                "Heartbeat, 2nd trimester", "wp17_structure_calibration", "detection well above the level after local calibration")
    if have("wp18_echo_retest"):
        r = load_run("wp18_echo_retest")[0]["score_agreement_all"]
        add("The echo expert gives the same answer for the same fetus scanned twice", EXPLORATORY, f"{r['n']} fetuses (2nd and 3rd trimester)",
            f"Spearman {num(r['rho'], 2)} {interval(r['ci'], 2)}", "Heartbeat", "wp18_echo_retest", "already falsified: the two scores do not agree")
    if have(BOUNDS_RUN):
        gate = load_run(BOUNDS_RUN)[0]["gate_G7"]
        add("False referrals of three experts for any dependence (Frechet range)", EXPLORATORY, "measured rates of each expert",
            f"{pct_interval(gate['range_at_measured_rates'])}; with uncertainty {pct_interval(gate['range_with_uncertainty'])}", "three datasets",
            BOUNDS_RUN, "a joint rate on paired patients outside the range")
    if have(TRADEOFF_RUN):
        smallest = load_run(TRADEOFF_RUN)[0]["rhythm_smallest_level"]
        add("With 14 calibration recordings the rhythm level acts as a multiple of 1/15", EXPLORATORY + " (arithmetic)",
            f"{smallest['n_healthy_calibration']} calibration recordings",
            f"effective level {num(smallest['effective_level_of_the_design'])}; effective ceiling {pct(smallest['effective_ceiling_of_the_design'])}",
            "NIFEADB", TRADEOFF_RUN, "none: it follows from the p-value formula")
    if have(RATE_RUN):
        c = load_run(RATE_RUN)[0]["comparison_with_original"]["logistic_rhythm"]
        diff = c["paired_difference_500hz_minus_original"]
        add("The rhythm expert does not rely on the sampling rate itself", EXPLORATORY, "26 recordings, all brought to 500 Hz",
            f"AUROC {num(c['auroc_500hz'])} {interval(c['auroc_500hz_ci'])}; change {num(diff['difference'])} {interval(diff['ci'])}", "NIFEADB", RATE_RUN,
            "AUROC falling after the common 500 Hz step; the recording setting itself is not removed by resampling")
    if have(SIZE_RUN):
        flag = load_run(SIZE_RUN)[0]["flag_642x506"]
        add("Calibrating inside each image-size group does not restore detection at 642x506", EXPLORATORY, f"{flag['n_abnormal']} defects at 642x506",
            f"caught {flag['group_conditional_caught']} of {flag['n_abnormal']}", "DELTA", SIZE_RUN, "defects caught there once calibrated in-group")
    if have(TRUST_BASELINE_RUN):
        swap = load_run(TRUST_BASELINE_RUN)[0]["by_women"]["test"]["swap"]["locked_minus_cross_correlation"]
        add("A plain pulse-train cross-correlation tells swapped pairs apart better than the locked share", EXPLORATORY, "test women",
            f"AUROC difference (locked minus cross-correlation) {num(swap['difference'])} {interval(swap['ci'])}", "NInFEA", TRUST_BASELINE_RUN,
            "the difference interval covering 0 on new recordings")
    if have(TRANSFER_RUN):
        m = load_run(TRANSFER_RUN)[0]
        holds = m["smallest_calibration_that_holds"]["local"]
        estimate = (f"about {holds['calibration_women']} women ({num(holds['calibration_recordings_median'], 0)} recordings)"
                    if isinstance(holds, dict) else holds)
        add("Local calibration keeps rhythm false alarms near the level from a modest number of healthy recordings", EXPLORATORY,
            f"{m['n_recordings_readable']} recordings of {m['n_women_readable']} women", estimate, "NIFEADB to NInFEA (one device pair)", TRANSFER_RUN,
            "another device pair needing far more; sensitivity after local calibration is unknown")
    if have(ALLOCATION_RUN):
        m = load_run(ALLOCATION_RUN)[0]
        s = m["selection_on_one_half_valued_on_the_other_at_design_budget"]
        key = "optimised (same ceiling) - design shares"
        add("Splitting the same false-referral ceiling differently adds true referrals", EXPLORATORY,
            "levels chosen on half the people, valued on the other half",
            f"{num(s[key + ' mean per 1000'], 2)} per 1000 [{num(s[key + ' interval low'], 2)}, {num(s[key + ' interval high'], 2)}]",
            "DELTA and NIFEADB, assumed prevalences", ALLOCATION_RUN, "an interval covering 0 (a gain too small to matter)")
    header = ["Claim", "Kind", "n", "Estimate [95% CI]", "Dataset", "Run", "Would be overturned by"]
    lead = ("Every claim of the project in one table, generated from the run folders. 'Confirmatory' rows were tested as planned before the held-out data "
            "were used; 'exploratory' rows were added after the results were known (sections 9 to 12) and their intervals are not corrected for the "
            "number of checks, so read them as exploratory only.")
    return "## Claim ledger\n\n" + lead + "\n\n" + table("t00_claim_ledger", header, rows, sorted(set(runs)))


def section_review_checks() -> str:
    """Section 12: the checks asked for in the review of 20 October 2026."""
    parts = ["## 12. Checks requested on 20 October 2026 (made after seeing the results)",
             "Each subsection re-uses saved outputs or refits only linear probes, as the original runs did; the settings were fixed in each run's "
             "config before computing. Intervals are not corrected for the number of checks."]
    if have(RATE_RUN):
        m = load_run(RATE_RUN)[0]
        rows = []
        for key, r in m["recording_level"].items():
            w = m["without_native_500hz_recordings"][key]
            c = m["comparison_with_original"].get(key)
            diff = c.get("paired_difference_500hz_minus_original") if c else None
            rows.append([key.replace("__", ", "), r.get("role", "interval expert"), f"{num(r['auroc'])} {interval(r['auroc_ci'])}",
                         f"{num(w['auroc'])} {interval(w['auroc_ci'])}", num(c["original_auroc_published"]) if c else "-",
                         f"{num(diff['difference'])} {interval(diff['ci'])}" if diff else ("not compared: saved scores tie" if c else "-")])
        unchanged = m["control_inputs_unchanged_from_wp4c"]
        notes = [f"**12.1 Rhythm expert with every recording at 500 Hz (run {RATE_RUN}).** The {m['dataset']['resampled_to_500hz']} recordings sampled at 1 kHz were "
                 "reduced to 500 Hz before the frozen extraction, features and probes. The four recordings already at 500 Hz got identical input, and their "
                 f"beats equal run wp4a's: {all(m['native_500hz_recordings_beats_identical_to_wp4a'].values())}. The control inputs were resampled per "
                 "recording here and per segment in wp4c, so their embeddings are " + ("identical" if all(unchanged.values()) else "close but not identical")
                 + f"; the mother's chest lead scores {num(m['comparison_with_original']['ecgfounder__maternal_chest']['auroc_500hz'])} in both runs. " + m["reading"]]
        parts += notes + [table("t12a_rhythm_500hz", ["Score", "Role", "AUROC, 26 recordings [95% CI]", "AUROC, 22 recordings from one setting [95% CI]",
                                                      "Original AUROC", "Change from the original [95% CI, paired]"], rows, [RATE_RUN])]
    if have(SIZE_RUN):
        m, provenance = load_run(SIZE_RUN)
        rows = []
        for e in m["by_group"]:
            auc = f"{num(e['auroc'])} {interval(e['auroc_ci'])}" if "auroc" in e else "-"
            rows.append([e["group"], e["n_normal"], e["n_abnormal"], auc, min(e["calibration_normals_per_fold"]), yes(e["can_alarm_in_every_fold"]),
                         f"{e['marginal_sensitivity']['k']}/{e['marginal_sensitivity']['n']}", f"{e['marginal_false_alarms']['k']}/{e['marginal_false_alarms']['n']}",
                         f"{e['group_conditional_sensitivity']['k']}/{e['group_conditional_sensitivity']['n']}",
                         f"{e['group_conditional_false_alarms']['k']}/{e['group_conditional_false_alarms']['n']}"])
        overall = m["overall"]
        parts += [f"**12.2 Structure expert by image size, calibrated inside each group (run {SIZE_RUN}).** The run's conformal p-values were recomputed "
                  f"from the saved embeddings and equal the published ones ({m['published_p_values_reproduced']}). Ranking each case only among the "
                  "calibration normals of its own size group keeps false alarms at the level inside every group, but a group needs at least "
                  f"{provenance['config']['minimum_calibration_normals_to_alarm']} calibration normals in a fold to alarm at all. Overall, group-conditional "
                  f"calibration caught {overall['group_conditional']['sensitivity']['k']} of {overall['group_conditional']['sensitivity']['n']} defects with "
                  f"{overall['group_conditional']['false_alarms']['k']} of {overall['group_conditional']['false_alarms']['n']} normal cases flagged (marginal: "
                  f"{overall['marginal']['sensitivity']['k']} and {overall['marginal']['false_alarms']['k']}). " + m["reading"],
                  table("t12b_structure_by_size_group_conditional",
                        ["Image size", "Normal", "Defect", "AUROC [95% CI]", "Calibration normals (smallest fold)", "Can alarm in every fold",
                         "Caught (marginal)", "Normal flagged (marginal)", "Caught (in-group)", "Normal flagged (in-group)"], rows, [SIZE_RUN])]
    if have(TRUST_BASELINE_RUN):
        m = load_run(TRUST_BASELINE_RUN)[0]
        rows = []
        for women in ("test", "all"):
            block = m["by_women"][women]
            for mismatch, part in (("the mother's beats", "maternal_substitution"), ("another woman's fetal ECG", "swap")):
                for score, value in block[part].items():
                    if "minus" in score:
                        rows.append([women, mismatch, score.replace("_", " "), "-", f"{num(value['difference'])} {interval(value['ci'])}", yes(value["can_be_ranked"])])
                    else:
                        rows.append([women, mismatch, score.replace("_", " "), f"{num(value['auroc'])} {interval(value['ci'])}", "-", "-"])
        parts += [f"**12.3 Baselines for the trust check (run {TRUST_BASELINE_RUN}).** The trust check is {m['wording']}. Heart rate alone nearly matches it "
                  "against the mother's beats, and a plain cross-correlation of the two beat trains separates swapped pairs better than the locked share. "
                  f"Gate G3 is unchanged ({m['gate_G3_unchanged']['note']}). Failure modes no recording here tests: " + "; ".join(m["untested_failure_modes"]) + ".",
                  table("t12c_trust_baselines", ["Women", "Mismatch", "Score", "AUROC [95% CI]", "Difference [95% CI, women resampled]", "Can be ranked"],
                        rows, [TRUST_BASELINE_RUN])]
    if have(TRANSFER_RUN):
        m, provenance = load_run(TRANSFER_RUN)
        rows = [[r["calibration_women"], num(r["calibration_recordings_median"], 0), yes(r["can_alarm_in_every_draw"]),
                 f"{pct(r['local_mean'])} [{pct(r['local_p5'])}, {pct(r['local_p95'])}]", pct_interval(r["beta_range_90_at_median_size"]),
                 f"{pct(r['pooled_mean'])} [{pct(r['pooled_p5'])}, {pct(r['pooled_p95'])}]"] for r in m["curve"]]
        shift, transported = m["shift"], m["transported"]
        smallest = m["smallest_calibration_that_holds"]
        scope = m["scope"][0].upper() + m["scope"][1:]

        def held(entry) -> str:
            return (f"{entry['calibration_women']} women ({num(entry['calibration_recordings_median'], 0)} recordings, median)"
                    if isinstance(entry, dict) else str(entry))

        parts += [f"**12.4 How many local healthy recordings the rhythm expert needs (run {TRANSFER_RUN}).** {scope}. Calibrated on the "
                  f"14 NIFEADB recordings it flagged {transported['flagged']} of {transported['n']} healthy NInFEA recordings ({pct(transported['false_alarm_rate'])}). "
                  f"Kolmogorov-Smirnov distance between the healthy scores of the two datasets {num(shift['distance'])} {interval(shift['ci'])}; the observed excess "
                  f"of false alarms, {pct(shift['observed_excess_false_alarms'])}, is within it, as theory says. Criterion: {provenance['config']['holds']} "
                  f"(level {provenance['config']['level']}, tolerance {provenance['config']['tolerance']}; revised after a trial run, see the run's config). Smallest "
                  f"calibration that holds: local {held(smallest['local'])}; pooled with the NIFEADB recordings {held(smallest['pooled'])}. Per-draw percentiles include the "
                  "noise of the few test women left; the Beta column is the spread expected from the calibration set alone.",
                  table("t12d_calibration_transfer", ["Calibration women", "Recordings (median)", "Can alarm in every draw",
                                                       "Local: mean false alarms [5th, 95th percentile]", "Expected from the calibration set alone (Beta, 90%)",
                                                       "Pooled with NIFEADB: mean [5th, 95th]"], rows, [TRANSFER_RUN])]
        figure, axis = plt.subplots(figsize=(8, 5))
        women = [r["calibration_women"] for r in m["curve"]]
        axis.fill_between(women, [r["local_p5"] for r in m["curve"]], [r["local_p95"] for r in m["curve"]], color="tab:blue", alpha=THIRD,
                          label="local calibration, 5th to 95th percentile of draws")
        axis.plot(women, [r["local_mean"] for r in m["curve"]], color="tab:blue", marker="o", label="local calibration, mean")
        axis.plot(women, [r["pooled_mean"] for r in m["curve"]], color="tab:orange", marker="s", label="pooled with the 14 NIFEADB recordings, mean")
        axis.axhline(transported["false_alarm_rate"], color="tab:red", linestyle="--", label="carried over from NIFEADB")
        axis.axhline(provenance["config"]["level"], color="black", linestyle=":", label="alarm level")
        axis.set_xlabel("Healthy NInFEA women used for calibration")
        axis.set_ylabel("Share of other women's healthy recordings flagged")
        axis.set_title("Rhythm false alarms against local calibration size")
        axis.legend(fontsize=7)
        parts.append(save(figure, "f10_calibration_transfer", [TRANSFER_RUN]))
    if have(ALLOCATION_RUN):
        m, provenance = load_run(ALLOCATION_RUN)
        design_budget = sum(m["design_levels"].values())
        rows = [[kind, f"{num(r['structure_level'])} / {num(r['function_level'])} / {num(r['rhythm_level'])}",
                 f"{num(r['structure_effective'])} / {num(r['function_effective'])} / {num(r['rhythm_effective'])}", num(r["effective_ceiling"]),
                 pct(r["structure_sensitivity"]), pct(r["rhythm_sensitivity"]), num(r["true_referrals_per_1000"], 2),
                 num(r["false_referrals_per_1000_ceiling"], 0),
                 f"{num(r['false_referrals_per_1000_measured_low'], 0)} to {num(r['false_referrals_per_1000_measured_high'], 0)}"]
                for kind, r in m["at_design_budget_base_prevalence"].items()]
        key = "optimised (same ceiling) - design shares"
        split_rows = []
        for raw in read_csv(ALLOCATION_RUN, "selection_split_half.csv"):
            if not np.isclose(float(raw["budget"]), design_budget):
                continue
            s = {k: float(v) for k, v in raw.items()}
            whole = "optimised - design shares"
            split_rows.append([num(s["rhythm_prevalence_factor"], 1),
                               f"{num(s[key + ' mean per 1000'], 2)} [{num(s[key + ' interval low'], 2)}, {num(s[key + ' interval high'], 2)}]",
                               pct(s[key + " share above 0"]),
                               f"{num(s[whole + ' mean per 1000'], 2)} [{num(s[whole + ' interval low'], 2)}, {num(s[whole + ' interval high'], 2)}]",
                               pct(s["share choosing another rhythm level at the same ceiling"])])
        size_rows = [[r["dependence"], r["tests"], num(r["total"], 2), num(r["bonferroni"], 4), num(r["holm"], 4), num(r["simes"], 4), r["simes_guaranteed"]]
                     for r in m["size_check"]]
        curve = read_csv(ALLOCATION_RUN, "decision_curve.csv")
        strategies = list(dict.fromkeys(r["strategy"] for r in curve))
        harms = list(dict.fromkeys(r["harm_ratio"] for r in curve))
        curve_rows = [[harm] + [num(float(next(r for r in curve if r["harm_ratio"] == harm and r["strategy"] == s)["net_benefit_per_1000_at_ceiling"]), 2)
                                for s in strategies] for harm in harms]
        headline = [[name, num(h["nominal_ceiling"]), num(h["effective_ceiling"]), pct_interval(h["measured_healthy_range_any_dependence"]),
                     ", ".join(f"{k} {pct(v)}" for k, v in h["sensitivity"].items()), num(h["true_referrals_per_1000"], 2), num(h["false_referrals_per_1000_ceiling"], 0)]
                    for name, h in m["headline_with_and_without_rhythm"].items()]
        illustration = m["simes_illustration_assuming_independent_healthy_other_tests"]
        parts += [f"**12.5 Splitting the false-referral ceiling between the experts (run {ALLOCATION_RUN}).** {m['reading']} Budget = sum of the effective "
                  f"levels. Priors: {levels_text(m['priors'])}; {m['priors_note']}. {m['function_note'].capitalize()}. Choosing levels to maximise expected true "
                  "referrals under a cap on false positives, with the cut-offs estimated on one part of the data and tested on another, is the method of "
                  "Rubin, Dudoit and van der Laan (2006); optimal weights for weighted Bonferroni are reviewed by Roeder and Wasserman (2009). Only the "
                  "discrete conformal levels and the prevalence weights are specific to this setting (docs/positioning.md). At the decision rule's priors the "
                  f"like-for-like optimum sets the rhythm level to {num(m['at_design_budget_base_prevalence']['optimised, same ceiling as design shares']['rhythm_effective'])}: "
                  "the result depends on the assumed prevalences (table below).",
                  table("t12e_allocation_at_design_budget", ["Allocation", "Levels set (structure / function / rhythm)", "Levels acting", "Effective ceiling",
                                                            "Structural defects caught", "Arrhythmias caught", "True referrals per 1000 (assumed prevalences)",
                                                            "False referrals per 1000 (ceiling)", "False referrals per 1000 (measured, any dependence)"],
                        rows, [ALLOCATION_RUN]),
                  "Levels chosen on half of the people with each finding and valued on the other half (200 splits), at the design budget, for each multiple of "
                  "the assumed rhythm prevalence:",
                  table("t12f_allocation_split_half", ["Rhythm prevalence factor", "Gain at the same ceiling, per 1000 [95% of splits]", "Splits with a gain",
                                                      "Gain using the whole budget, per 1000 [95% of splits]", "Splits choosing another rhythm level"],
                        split_rows, [ALLOCATION_RUN]),
                  "Size of the yes/no referral under chosen dependence (simulated healthy people). For the referral decision Holm refers exactly whom "
                  "Bonferroni refers (Holm was run on the first 20000 simulated people of each row, where it referred the same people as Bonferroni: "
                  + yes(all(r["holm_equals_bonferroni_on_first_20000"] for r in m["size_check"])) + "); Simes is guaranteed only under positive dependence "
                  "and exceeds its total for a dependence built against it. "
                  f"Under the assumption that the two other tests behave like independent healthy tests, Simes would refer {pct(illustration['structure']['simes'])} "
                  f"of the defects against {pct(illustration['structure']['bonferroni_any_alarm'])} for Bonferroni with equal levels "
                  f"(own test alone {pct(illustration['structure']['own_test_only'])}); the gap is chance alarms of the other tests, not detection.",
                  table("t12g_size_check", ["Dependence", "Tests", "Total", "Bonferroni", "Holm", "Simes", "Simes guaranteed"], size_rows, [ALLOCATION_RUN]),
                  "Net benefit per 1000 people screened (true referrals minus the harm ratio times false referrals, false referrals at the ceiling). The harm "
                  "ratio - harm of one unnecessary referral relative to the benefit of one needed referral - is a clinical judgement not made here:",
                  table("t12h_net_benefit", ["Harm ratio"] + strategies, curve_rows, [ALLOCATION_RUN]),
                  "The headline with and without the rhythm expert:",
                  table("t12i_with_without_rhythm", ["Experts", "Nominal ceiling", "Effective ceiling", "Measured healthy range (any dependence)", "Caught",
                                                    "True referrals per 1000", "False referrals per 1000 (ceiling)"], headline, [ALLOCATION_RUN])]
    if have(THEORY_RUN):
        m = load_run(THEORY_RUN)[0]
        a = [r for r in read_csv(THEORY_RUN, "a_conformal_validity.csv") if r["scores"] == "normal"]
        rows = [[r["calibration_n"], r["level"], num(float(r["effective_level"]), 4), num(float(r["mean_rate"]), 4),
                 f"[{num(float(r['p5_rate']), 3)}, {num(float(r['p95_rate']), 3)}]", f"[{num(float(r['beta_p5']), 3)}, {num(float(r['beta_p95']), 3)}]"] for r in a]
        ends = m["b_ends_reached"]
        parts += [f"**12.6 The standard guarantees, simulated (run {THEORY_RUN}, synthetic data).** {m['reading'].capitalize()}. Mean false-alarm rate never above "
                  f"the effective level by more than 3 standard errors: {m['a_mean_rate_never_above_effective_level_by_more_than_3_se']}; largest gap between "
                  f"simulated and Beta percentiles {num(m['a_largest_gap_between_simulated_and_beta_percentiles'], 3)}. Union bound: every dependence inside "
                  f"{pct_interval(ends['range'])}: {m['b_every_dependence_inside_the_range']}, with the lower end reached by nested alarms ({pct(ends['lower'])}) and "
                  f"the upper by alarms that never come together ({pct(ends['upper'])}). Under a location shift the false-alarm rate stayed within the level "
                  f"plus the Kolmogorov-Smirnov distance: {m['c_every_shift_within_level_plus_ks']}. References: docs/theory_notes.md.",
                  table("t12j_conformal_validity", ["Calibration n", "Level", "Effective level", "Mean false-alarm rate", "Per-set rate, 5th-95th percentile",
                                                   "Beta law, 5th-95th percentile"], rows, [THEORY_RUN])]
    return "\n\n".join(parts) if len(parts) > 2 else ""


EXTERNAL = ("heartbeat_2T", "heartbeat_2T_test", "heartbeat_3T", "cardium", "cardium_grey")
EXTERNAL_NAMES = {"heartbeat_2T": "Heartbeat, 2nd trimester (primary)", "heartbeat_2T_test": "Heartbeat, 2nd-trimester test set",
                  "heartbeat_3T": "Heartbeat, 3rd trimester", "cardium": "CARDIUM, all images", "cardium_grey": "CARDIUM, grey images only"}


def shortcut_strength(values: dict) -> float:
    """A single-number shortcut can separate the classes in either direction."""
    return max(max(v, 1 - v) for v in values.values()) if values else float("nan")


def external_verdict(run: str) -> dict:
    """The reading rule fixed in docs/preregistration_external_echo.md (with its clarification of 27 Oct 2026)."""
    metrics, provenance = load_run(run)
    t = metrics["transported_calibration"]
    strongest = shortcut_strength(metrics["shortcut_single_numbers_auroc"])
    above_chance = t["auroc_ci"][0] > HALF
    beats_shortcuts = t["auroc"] > strongest
    level = provenance["config"]["alpha"]
    holds = t["false_alarms"]["ci"][0] <= level <= t["false_alarms"]["ci"][1]
    reasons = ([] if above_chance else ["its AUROC is not distinguishable from 0.5"]) +               ([] if beats_shortcuts else ["a single image number separates the classes at least as well"])
    return {"reads": above_chance and beats_shortcuts, "strongest_shortcut": strongest, "calibration_holds": holds, "level": level,
            "reasons": reasons}


def section_external_echo() -> str:
    if not have("wp11_external_echo_heartbeat_2T"):
        return ""
    parts = ["## 13. External echo data: Heartbeat and CARDIUM",
             "Both datasets come from the BCV group of Universidad de los Andes (Colombia) and were received through the authors' request "
             "forms (data cards: docs/data_cards/heartbeat.md, cardium.md). Their terms forbid redistribution, so only aggregate numbers "
             "appear here; per-patient outputs stay in each run's private folder. The frozen DELTA structure expert on these data is the "
             "test preregistered on 20 October 2026 (docs/preregistration_external_echo.md, with clarifications dated 27 October, made "
             "before any model saw the images); everything after 13.2 is exploratory."]
    if have("wp14_echo_overlap"):
        m = load_run("wp14_echo_overlap")[0]
        rows = [[pair.replace("__", " vs "), f"{v['images'][0]} / {v['images'][1]}", v["exact_image_pairs"], v["near_image_pairs"],
                 v["patient_pairs_exact"]] for pair, v in m["between_datasets"].items()]
        text = ["**13.1 Shared images and patients (runs wp14_echo_overlap, wp14b_echo_overlap_scan).** Whole frames compared by identical "
                "pixels and by a difference hash:", table("t13a_overlap_frames", ["Datasets", "Images", "Identical frames", "Near-identical frames",
                                                                                  "Patients sharing a frame"], rows, ["wp14_echo_overlap"])]
        if have("wp14b_echo_overlap_scan"):
            s = load_run("wp14b_echo_overlap_scan")[0]
            repeats = "; ".join(f"{name}: {r['images_repeated_across_patients']} images repeated across {r['patients_involved']} patients "
                                f"({r['with_different_labels']} with different labels)" for name, r in s["repeats_inside_datasets"].items())
            text.append(f"Heartbeat and CARDIUM store frames of different sizes, so run wp14b compared the cleaned scan regions instead: "
                        f"{s['candidate_image_pairs']} candidate image pairs, {s['matched_image_pairs']} confirmed by thumbnail correlation; "
                        f"{s['patient_pairs_same']} Heartbeat-CARDIUM patient pairs share two or more scans and {s['patient_pairs_single_image']} "
                        f"share one ({s['labels_agree_in_same_pairs']} of the {s['patient_pairs_same']} with the same label), so CARDIUM is largely "
                        f"a second export of Heartbeat's fetuses, not an independent dataset. Inside the datasets: {repeats}.")
        parts += text
    rows, verdicts = [], {}
    for name in EXTERNAL:
        run = f"wp11_external_echo_{name}"
        if not have(run):
            continue
        m = load_run(run)[0]
        t, local = m["transported_calibration"], m["local_calibration"]
        verdicts[name] = external_verdict(run)
        short = m["shortcut_single_numbers_auroc"]
        rows.append([EXTERNAL_NAMES[name], f"{m['cases']} ({t['n_abnormal']})", f"{num(t['auroc'])} {interval(t['auroc_ci'])}",
                     f"{t['sensitivity']['k']}/{t['sensitivity']['n']} ({pct(t['sensitivity']['share'])}) {pct_interval(t['sensitivity']['ci'])}",
                     f"{t['false_alarms']['k']}/{t['false_alarms']['n']} ({pct(t['false_alarms']['share'])}) {pct_interval(t['false_alarms']['ci'])}",
                     f"{local['sensitivity']['k']}/{local['sensitivity']['n']} ({pct(local['sensitivity']['share'])}) {pct_interval(local['sensitivity']['ci'])}",
                     f"{local['false_alarms']['k']}/{local['false_alarms']['n']} ({pct(local['false_alarms']['share'])}) {pct_interval(local['false_alarms']['ci'])}",
                     ", ".join(f"{k.replace('_', ' ')} {num(v)}" for k, v in short.items()),
                     f"{m['unusable_images_skipped']} / {m['cases_without_a_usable_image']}"])
    primary = verdicts.get("heartbeat_2T")
    reading = ""
    if primary:
        reading = ("**Preregistered reading, primary dataset:** " + ("supports 'reads heart structure'" if primary["reads"] else
                   "does not support 'reads heart structure' (" + " and ".join(primary["reasons"]) + "), which the fixed rule counts as "
                   "'shortcut'") + f". The rule asks for an AUROC interval above 0.5 and an AUROC above the strongest single-number "
                   f"shortcut ({num(primary['strongest_shortcut'])} here). The transported alarm level " +
                   ("holds" if primary["calibration_holds"] else "does not hold") +
                   f" (its false-alarm interval {'includes' if primary['calibration_holds'] else 'excludes'} the level {primary['level']}). "
                   "Secondary datasets, same rule: " + "; ".join(f"{EXTERNAL_NAMES[n]}: {'reads structure' if v['reads'] else 'shortcut'}, "
                                                                 f"calibration {'holds' if v['calibration_holds'] else 'does not hold'}"
                                                                 for n, v in verdicts.items() if n != "heartbeat_2T") + ".")
    parts += ["**13.2 The frozen DELTA structure expert on the new data (preregistered; runs wp11_external_echo_*).** Nothing was refitted. "
              "Alarms at level 0.05 with the DELTA calibration (transported) and with half of each dataset's normal cases (local). The "
              "single-number shortcut scores (image width, height, mean brightness) are AUROCs of those numbers alone.",
              table("t13b_external_test", ["Dataset", "Patients (CHD)", "AUROC [95% CI]", "Transported: CHD caught [CI]",
                                           "Transported: normal flagged [CI]", "Local: CHD caught [CI]", "Local: normal flagged [CI]",
                                           "Shortcut scores (AUROC)", "Unusable images / patients"], rows,
                    [f"wp11_external_echo_{n}" for n in verdicts]), reading]
    if have("wp15_reference_models"):
        m = load_run("wp15_reference_models")[0]
        rows = []
        for key, value in m["results"].items():
            arch, cohort, split = key.split("|")
            rows.append([arch, cohort, split, sum(value["n"]) // len(value["n"]), f"{num(value['auroc']['mean'])} ± {num(value['auroc']['sd_population'])}",
                         f"{pct(value['sensitivity']['mean'])} ± {pct(value['sensitivity']['sd_population'])}",
                         f"{pct(value['specificity']['mean'])} ± {pct(value['specificity']['sd_population'])}",
                         f"{pct(value['f1']['mean'])} ± {pct(value['f1']['sd_population'])}"])
        diff = m["heart_vit_2T_test_minus_reported_points"]
        parts += [f"**13.3 The authors' released models (run wp15_reference_models; exploratory).** Heart-ViT and five ImageNet-initialised "
                  f"baselines, loaded strictly from the released checkpoints and run with the authors' own inference and metric code on their "
                  f"splits. Heart-ViT on the 2nd-trimester test set minus the authors' README, in percentage points: "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in diff.items()) + f"; gate G5's 'reference model reproduced' is "
                  f"**{state(m['gate_G5_reference_model_reproduced'])}** (tolerance 0.5 points). '3T-unseen' applies each 2nd-trimester fold model "
                  f"to the {m['unseen_3T_patients']} 3rd-trimester patients never seen in the 2nd-trimester data. Mean ± SD over the 4 fold models.",
                  table("t13c_reference_models", ["Model", "Patients", "Split", "n per fold", "AUROC", "Sensitivity", "Specificity", "F1"],
                        [[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows], ["wp15_reference_models"])]
    probe_rows = []
    for name, run in (("Heartbeat 2T", "wp5_structure_heartbeat_2T"), ("Heartbeat 3T", "wp5_structure_heartbeat_3T"),
                      ("CARDIUM", "wp5_structure_cardium"), ("DELTA (for comparison)", "wp5_structure_delta")):
        if not have(run):
            continue
        metrics_run = load_run(run)[0]
        probes = metrics_run["probes"]
        same = metrics_run.get("same_image_size", {})
        inside = (f"{same['size']} ({same['cases']} cases): {num(same['probes']['fetalclip_scan']['auroc'])} / "
                  f"{num(same['probes']['no_anatomy_numbers']['auroc'])}" if same.get("probes") else "-")
        probe_rows.append([name] + [f"{num(probes[p]['auroc'])} {interval(probes[p]['ci'])}" for p in
                                    ("fetalclip_scan", "random_weights_scan", "fetalclip_overlay_only", "no_anatomy_numbers")] + [inside])
    if len(probe_rows) > 1:
        parts += ["**13.4 A FetalCLIP probe fitted on each dataset, with the same controls as DELTA (runs wp5_structure_*; exploratory).** "
                  "Case-level folds inside each dataset; the controls are the same network with random weights, the parts removed by the "
                  "cleaning step, and six numbers that describe the image rather than its content.",
                  table("t13d_probes_per_dataset", ["Dataset", "FetalCLIP, cleaned scan", "Random weights", "Removed parts only", "Six image numbers",
                                                    "Commonest image size: scan / six numbers"],
                        probe_rows, ["wp5_structure_heartbeat_2T", "wp5_structure_heartbeat_3T", "wp5_structure_cardium", "wp5_structure_delta"])]
    if have("wp16_structure_transfer"):
        m = load_run("wp16_structure_transfer")[0]
        rows = [[pair.replace("->", " to "), f"{num(v['auroc'])} {interval(v['ci'])}", v["n"], v["n_chd"], v["left_out_seen"]]
                for pair, v in m["transfer"].items()]
        shift_rows = [[name, num(v["val_auroc_mean"]), num(v["unseen_3T_auroc_mean"]), num(v["drop"])] for name, v in m["trimester_shift"].items()]
        parts += ["**13.5 Probes moved between datasets, and from the 2nd to the 3rd trimester (run wp16_structure_transfer; exploratory).** "
                  "Fitted on every case of one dataset, applied to another; target patients seen in the source are left out. Within-dataset AUROC: "
                  + ", ".join(f"{k} {num(v)}" for k, v in m["within_dataset_auroc"].items()) + ".",
                  table("t13e_transfer", ["Fitted on, applied to", "AUROC [95% CI]", "Patients", "CHD", "Left out (seen in the source)"], rows,
                        ["wp16_structure_transfer"]),
                  f"Trimester shift (hypothesis H6): mean AUROC on the authors' 2nd-trimester validation folds and on the {m['unseen_3T_patients']} "
                  "3rd-trimester patients never seen in 2nd-trimester data, for the FetalCLIP probe and the authors' models.",
                  table("t13f_trimester_shift", ["Model", "2T validation AUROC", "Unseen 3T AUROC", "Drop"], shift_rows, ["wp16_structure_transfer"])]
    if have("wp17_structure_calibration"):
        m = load_run("wp17_structure_calibration")[0]
        rows = []
        for name, d in m["by_dataset"].items():
            for r in d["curve"]:
                lf, ld = r.get("local_false_alarms"), r.get("local_detection")
                rows.append([EXTERNAL_NAMES.get(name, name), r["local_healthy"], yes(r["can_alarm_local"]),
                             f"{pct(lf['mean'])} [{pct(lf['p5'])}, {pct(lf['p95'])}]" if lf else "-",
                             f"{pct(ld['mean'])} [{pct(ld['p5'])}, {pct(ld['p95'])}]" if ld else "-",
                             f"{pct(r['pooled_false_alarms']['mean'])}", f"{pct(r['pooled_detection']['mean'])}"])
        summary = "; ".join(f"{EXTERNAL_NAMES.get(n, n)}: transported calibration flagged {pct(d['transported']['false_alarms'])} of normal "
                            f"fetuses and caught {pct(d['transported']['detection'])} of CHD; local calibration holds from "
                            + (f"{d['smallest_that_holds']['local']['healthy']} healthy patients, where it catches "
                               f"{pct(d['smallest_that_holds']['local']['detection_mean'])} of CHD"
                               if isinstance(d['smallest_that_holds']['local'], dict) else str(d['smallest_that_holds']['local']))
                            + f"; KS distance {num(d['shift']['distance'])} {interval(d['shift']['ci'])}"
                            for n, d in m["by_dataset"].items())
        parts += ["**13.6 Does the structure expert's alarm level travel, and what does local calibration cost in detection? (run "
                  "wp17_structure_calibration; exploratory).** " + summary + ".",
                  table("t13g_structure_calibration", ["Dataset", "Local healthy patients", "Can alarm", "Local: normal flagged [5th, 95th]",
                                                       "Local: CHD caught [5th, 95th]", "Pooled with DELTA: normal flagged", "Pooled: CHD caught"],
                        rows, ["wp17_structure_calibration"])]
    if have("wp18_echo_retest"):
        m = load_run("wp18_echo_retest")[0]
        a = m["score_agreement_all"]
        parts.append(f"**13.7 The same fetus scanned twice (run wp18_echo_retest; exploratory).** {m['patients_in_both_trimesters']} Heartbeat "
                     f"patients were scanned in both trimesters ({m['label_differs_between_trimesters']} with different labels, left out). Spearman "
                     f"correlation of the frozen expert's two scores {num(a['rho'], 2)} {interval(a['ci'], 2)} (n {a['n']}); alarms with the "
                     f"transported calibration agree with kappa {num(m['alarm_kappa_transported'], 2)} (table: {m['alarm_table_transported']}).")
    return "\n\n".join(parts)


def verdict_fallback(gate: dict) -> str:
    return "cannot be evaluated: " + gate["reason"] if "reason" in gate else state(gate.get("passed"))


def section_gates() -> str:
    labelled = structure_runs()
    entries = [("G1", "wp1_cinc2013", "gate_G1"), ("G2", "wp2_doppler_cycles", "gate_G2"), ("G3", "wp3_trust_check", "gate_G3"),
               ("G4", "wp4c_foundation", "gate_G4"), ("G5", labelled[0] if labelled else None, "gate_G5"), ("G6", "wp6_fusion", "gate_G6")]
    if (RUNS / BOUNDS_RUN / "metrics.json").exists():
        entries.append(("G7", BOUNDS_RUN, "gate_G7"))
    rows = []
    for name, run, key in entries:
        if run is None:
            rows.append([name, "cannot be evaluated: no labelled echo data", GATE_RULES.get(name, ""), "-", "-", "-"])
            continue
        metrics, provenance = load_run(run)
        gate = metrics[key]
        if name == "G6":
            verdict = "decided by the property tests (python -m pytest tests/test_fusion.py)"
        elif name == "G7":
            verdict = (f"at the measured rates {state(gate['passed_at_measured_rates'])} ({pct_interval(gate['range_at_measured_rates'])}); "
                       f"with uncertainty {state(gate['passed_with_uncertainty'])} ({pct_interval(gate['range_with_uncertainty'])}); gate defined after seeing the results")
        elif name == "G5" and have("wp15_reference_models", "wp5_structure_heartbeat_2T"):
            reproduced = load_run("wp15_reference_models")[0]["gate_G5_reference_model_reproduced"]
            verdict = ((f"{'met' if reproduced else 'partly met'} on Heartbeat: the released Heart-ViT {'was' if reproduced else 'was NOT'} "
                        "reproduced (run wp15_reference_models), and patient-level AUROC with an interval and the shortcut audit are reported "
                        "(run wp5_structure_heartbeat_2T); on DELTA no released model exists")
                       if gate.get("auroc_with_interval") else verdict_fallback(gate))
        elif name == "G5" and gate.get("auroc_with_interval"):
            verdict = "partly met: patient-level AUROC with an interval and the shortcut audit are reported; " + gate["reason"].split(";")[0]
        elif "reason" in gate:
            verdict = "cannot be evaluated: " + gate["reason"]
        else:
            verdict = state(gate.get("passed"))
        rows.append([name, verdict, gate.get("rule", GATE_RULES.get(name, "")), run, provenance["code_sha256"][:12], provenance["created_utc"]])
    return "## Gates\n\n" + table("t0_gates", ["Gate", "Outcome", "Rule", "Run", "Code hash", "Run time (UTC)"], rows, [e[1] for e in entries if e[1]])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Results of the rebuild\n\nGenerated by `scripts/build_report.py` from the run folders under `runs/`. "
        "Do not edit by hand: change the analysis, re-run it, and rebuild. Tables are also saved as CSV in `reports/tables/`, "
        "figures in `reports/figures/`, and `reports/manifest.json` lists the runs behind every file.",
        claim_ledger(), section_gates(), section_wp1(), section_wp2(), section_wp3(), section_wp2b(), section_wp4(), section_wp5(), section_wp6(),
        section_wp6b(), section_guarantees(), section_checks(), section_review_checks(), section_external_echo(), section_threats(),
    ]
    (REPORTS / "RESULTS.md").write_text("\n\n".join(part for part in parts if part) + "\n", encoding="utf-8")
    MANIFEST.append({"file": "RESULTS.md", "runs": sorted({run for entry in MANIFEST for run in entry["runs"]})})
    (REPORTS / "manifest.json").write_text(json.dumps({"outputs": MANIFEST}, indent=2), encoding="utf-8")
    print(f"wrote {REPORTS / 'RESULTS.md'}, {len([m for m in MANIFEST if m['file'].startswith('tables/')])} tables, "
          f"{len([m for m in MANIFEST if m['file'].startswith('figures/')])} figures")


if __name__ == "__main__":
    main()
