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
    return "\n\n".join(["## 5. Rhythm expert (NIFEADB, 12 arrhythmia and 14 normal)", *notes, text, leak_text, probe_text, check_text, picture])


PROBE_NAMES = {
    "fetalclip_scan": "FetalCLIP, de-identified scan (the structure expert)",
    "fetalclip_unaltered_image": "FetalCLIP, image as released",
    "random_weights_scan": "same network with random weights (control)",
    "fetalclip_overlay_only": "FetalCLIP, overlays only (shortcut audit)",
    "no_anatomy_numbers": "six numbers without anatomy (shortcut audit)",
}


def structure_runs() -> list[str]:
    return sorted(path.name for path in RUNS.glob("wp5_structure_*") if (path / "metrics.json").exists())


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
    run = "wp5_echo_readiness"
    metrics, _ = load_run(run)
    clip = metrics["fetalclip"]
    labelled = structure_runs()
    readiness = [f"Readiness check on {metrics['frames']} unlabelled frames ({metrics['cases']} folders), read in memory only: the de-identification tool kept a median of "
                 f"{pct(metrics['deidentification']['scan_share_of_frame_median'], 0)} of each frame and blanked everything else; "
                 f"not verified: {metrics['deidentification']['not_verified']}.",
                 f"{clip['checkpoint']['claimed_name']} (SHA-256 {clip['checkpoint']['sha256'][:16]}..., {clip['checkpoint']['n_parameters']:,} parameters, "
                 f"licence {clip['checkpoint']['licence']}) loaded and produced embeddings of shape {clip['embedding_shape']}."]
    controls = [path.name for path in RUNS.glob("wp5b_structure_controls_*") if (path / "metrics.json").exists()]
    MANIFEST.append({"file": "RESULTS.md#structure", "runs": [run, *labelled, *controls]})
    if not labelled:
        lines = [f"**{metrics['statement']}**", metrics["blocked"], *readiness, f"Gate G5: **cannot be evaluated** - {metrics['gate_G5']['reason']}."]
        return "\n\n".join(["## 6. Structure expert (fetal echo)", *lines])
    lines = ["Gate G5 (the released reference model reproduced; pooled patient-level AUROC with an interval; shortcut audit reported): "
             "**partly met** - the AUROC with its interval and the shortcut audit are reported; no released reference model exists for the dataset used. "
             "**The shortcut audit finds label information outside the heart**: probes that see little or no cardiac anatomy separate the classes well above chance (tables below). "
             "The structure expert's AUROC is therefore an upper bound on what heart structure contributes, not a measure of it.",
             *[section_structure(name) for name in labelled],
             "### Readiness check made before labelled data was available", *readiness]
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


def section_gates() -> str:
    labelled = structure_runs()
    entries = [("G1", "wp1_cinc2013", "gate_G1"), ("G2", "wp2_doppler_cycles", "gate_G2"), ("G3", "wp3_trust_check", "gate_G3"),
               ("G4", "wp4c_foundation", "gate_G4"), ("G5", labelled[0] if labelled else "wp5_echo_readiness", "gate_G5"), ("G6", "wp6_fusion", "gate_G6")]
    rows = []
    for name, run, key in entries:
        metrics, provenance = load_run(run)
        gate = metrics[key]
        if name == "G6":
            verdict = "decided by the property tests (python -m pytest tests/test_fusion.py)"
        elif name == "G5" and gate.get("auroc_with_interval"):
            verdict = "partly met: patient-level AUROC with an interval and the shortcut audit are reported; " + gate["reason"].split(";")[0]
        elif "reason" in gate:
            verdict = "cannot be evaluated: " + gate["reason"]
        else:
            verdict = state(gate.get("passed"))
        rows.append([name, verdict, gate.get("rule", GATE_RULES.get(name, "")), run, provenance["code_sha256"][:12], provenance["created_utc"]])
    return "## Gates\n\n" + table("t0_gates", ["Gate", "Outcome", "Rule", "Run", "Code hash", "Run time (UTC)"], rows, [e[1] for e in entries])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Results of the rebuild\n\nGenerated by `scripts/build_report.py` from the run folders under `runs/`. "
        "Do not edit by hand: change the analysis, re-run it, and rebuild. Tables are also saved as CSV in `reports/tables/`, "
        "figures in `reports/figures/`, and `reports/manifest.json` lists the runs behind every file.",
        section_gates(), section_wp1(), section_wp2(), section_wp3(), section_wp2b(), section_wp4(), section_wp5(), section_wp6(), section_wp6b(),
    ]
    (REPORTS / "RESULTS.md").write_text("\n\n".join(part for part in parts if part) + "\n", encoding="utf-8")
    MANIFEST.append({"file": "RESULTS.md", "runs": sorted({run for entry in MANIFEST for run in entry["runs"]})})
    (REPORTS / "manifest.json").write_text(json.dumps({"outputs": MANIFEST}, indent=2), encoding="utf-8")
    print(f"wrote {REPORTS / 'RESULTS.md'}, {len([m for m in MANIFEST if m['file'].startswith('tables/')])} tables, "
          f"{len([m for m in MANIFEST if m['file'].startswith('figures/')])} figures")


if __name__ == "__main__":
    main()
