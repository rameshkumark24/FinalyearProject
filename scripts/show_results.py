"""Print the main results in the terminal, read from the run folders (nothing typed by hand).

    python scripts/show_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import load_run  # noqa: E402

LINE = "=" * 78


def m(run: str) -> dict:
    return load_run(run)[0]


def ci(pair, digits: int = 3) -> str:
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    print(LINE)
    print(" Screening the fetal heart from unpaired tests (Structure - Function - Rhythm)")
    print(" Every number below is read from runs/<run>/metrics.json")
    print(LINE)

    section("Gates (targets fixed before the held-out runs; G7 added after a review)")
    g1, g2, g3 = m("wp1_cinc2013")["gate_G1"], m("wp2_doppler_cycles")["gate_G2"], m("wp3_trust_check")["gate_G3"]
    g4, g5, g7 = m("wp4c_foundation")["gate_G4"], m("wp5_structure_delta")["gate_G5"], m("wp7_dependence_bounds")["gate_G7"]
    state = lambda ok: "MET" if ok else "NOT MET"  # noqa: E731
    print(f"  G1 fetal beat detection     {state(g1['passed']):8s} median F1 {g1['median_f1_test']:.3f}")
    print(f"  G2 Doppler cycles           {state(g2['passed']):8s} median F1 {g2['median_f1_test']:.3f}")
    print(f"  G3 ECG-Doppler trust check  {state(g3['passed']):8s} AUROC vs mother's beats {g3['auroc_maternal_test']:.3f}; "
          f"agreement limits {ci(g3['beat_level_limits_bpm'], 1)} bpm (target +/-10)")
    print(f"  G4 rhythm expert            {state(g4['passed']):8s}")
    print(f"  G5 structure expert         PARTLY MET  ({g5['reason']})")
    print(f"  G6 decision rule            MET by the property tests")
    print(f"  G7 bounds without pairs     {state(g7['passed_at_measured_rates'])} at measured rates ({pct(g7['range_at_measured_rates'][0])} to {pct(g7['range_at_measured_rates'][1])}), "
          f"{state(g7['passed_with_uncertainty'])} with uncertainty ({pct(g7['range_with_uncertainty'][0])} to {pct(g7['range_with_uncertainty'][1])})")

    section("Function expert (Doppler timing, 60 healthy NInFEA recordings)")
    f = m("wp2b_function_reference")["healthy_recordings_flagged"]
    for alpha, row in f.items():
        print(f"  alarm level {alpha}: {row['recordings_flagged']} of {row['recordings']} healthy recordings flagged")

    section("Rhythm expert (NIFEADB: 12 arrhythmia, 14 normal, leave one subject out)")
    rec, single = m("wp4b_rhythm_intervals")["recording_level"], m("wp4b_rhythm_intervals")["single_index"]
    probe = m("wp4c_foundation")["arrhythmia_probe_nifeadb"]
    print(f"  logistic regression (the rule's expert) AUROC {rec['logistic_rhythm']['auroc']:.3f} {ci(rec['logistic_rhythm']['auroc_ci'])}")
    print(f"  one interval index, nothing fitted      AUROC {single['succ_rel_median']['auroc']:.3f} {ci(single['succ_rel_median']['auroc_ci'])}")
    print(f"  ECGFounder (frozen, hash-checked)       AUROC {probe['ecgfounder__fetal']['auroc']:.3f} {ci(probe['ecgfounder__fetal']['auroc_ci'])}")
    print(f"  same network, random weights            AUROC {probe['random_init__fetal_stretch2']['auroc']:.3f} "
          f"{ci(probe['random_init__fetal_stretch2']['auroc_ci'])}")
    with open(RUNS / "wp9_checks_after_results" / "rhythm_1khz_only.csv", encoding="utf-8") as handle:
        row = next(r for r in csv.DictReader(handle) if r["score"] == "oof_logistic_rhythm")
    print(f"  check after a review: logistic on the 1 kHz recordings only AUROC {float(row['auroc_1khz']):.3f} "
          f"[{float(row['ci_1khz_low']):.3f}, {float(row['ci_1khz_high']):.3f}]")

    section("Structure expert (DELTA echo: 752 cases, 133 with a defect)")
    s = m("wp5_structure_delta")
    probes = s["probes"]
    print(f"  FetalCLIP linear probe                  AUROC {probes['fetalclip_scan']['auroc']:.3f} {ci(probes['fetalclip_scan']['ci'])}")
    print(f"  on the authors' own test split          AUROC {s['own_partition']['fetalclip_scan']['auroc']:.3f}")
    print(f"  controls with little or no heart anatomy: removed parts {probes['fetalclip_overlay_only']['auroc']:.3f}, "
          f"six image numbers {probes['no_anatomy_numbers']['auroc']:.3f}  -> AUROC is an upper bound")
    alarm = s["alarm"]
    print(f"  at alarm level {alarm['alpha']}: caught {alarm['sensitivity']['caught']} of {alarm['sensitivity']['n']} defects, "
          f"flagged {alarm['false_alarms']['flagged']} of {alarm['false_alarms']['n']} normal cases")

    section("Decision rule on 60 real ECG-Doppler pairs of healthy fetuses (echo assumed normal)")
    b = m("wp6_fusion")["B_real_paired_cohort"]
    local = b["with_rhythm_score_calibrated_on_ninfea_leave_one_woman_out"]["false_referrals"]
    print(f"  rhythm expert calibrated on another dataset: {pct(b['false_referrals']['share'])} falsely referred (ceiling 15%)")
    print(f"  rhythm expert calibrated locally:            {pct(local['share'])} falsely referred (ceiling 15%)")

    section("What holds without paired patients (checks added after a review)")
    print(f"  false referrals for any dependence between the tests: {pct(g7['range_at_measured_rates'][0])} to "
          f"{pct(g7['range_at_measured_rates'][1])} (ceiling {pct(g7['ceiling'])})")
    print(f"  with the uncertainty of each expert's rate:            {pct(g7['range_with_uncertainty'][0])} to {pct(g7['range_with_uncertainty'][1])}")
    print(f"  ceiling the design actually guarantees:                {pct(m('wp8_tradeoff')['rhythm_smallest_level']['effective_ceiling_of_the_design'])}")
    print("  not measured: any rate on patients with all three tests")
    print("  these are standard bounds (union / Frechet), not new methods: docs/positioning.md")
    print("  combining tests widens what is checked and adds false referrals; it does not detect any one condition better")

    if (RUNS / "wp10_allocation" / "metrics.json").exists():
        section("Checks requested in the review of 20 Oct 2026 (made after seeing the results)")
        rate = m("wp4e_rhythm_500hz")
        c = rate["comparison_with_original"]["logistic_rhythm"]
        w = rate["without_native_500hz_recordings"]["logistic_rhythm"]
        print(f"  rhythm expert, every recording at 500 Hz: AUROC {c['auroc_500hz']:.3f} {ci(c['auroc_500hz_ci'])}; "
              f"22 recordings from one setting {w['auroc']:.3f} {ci(w['auroc_ci'])}")
        print(f"  mother's chest lead (no fetal signal) still scores {rate['recording_level']['ecgfounder__maternal_chest']['auroc']:.3f}: "
              "the four 500 Hz recordings differ in another way")
        flag = m("wp5c_structure_by_size")["flag_642x506"]
        print(f"  structure expert in 642x506 images: AUROC {flag['auroc']:.3f} {ci(flag['auroc_ci'])}, caught "
              f"{flag['group_conditional_caught']} of {flag['n_abnormal']} with in-group calibration")
        trust = m("wp3b_trust_baselines")["by_women"]["test"]
        print(f"  trust check vs mother's beats: locked share {trust['maternal_substitution']['locked_share']['auroc']:.3f}, "
              f"heart rate alone {trust['maternal_substitution']['heart_rate_only']['auroc']:.3f}")
        print(f"  trust check vs another woman's ECG: locked share {trust['swap']['locked_share']['auroc']:.3f}, "
              f"beat-train cross-correlation {trust['swap']['cross_correlation']['auroc']:.3f}")
        transfer = m("wp13_calibration_transfer")
        print(f"  rhythm false alarms carried over from NIFEADB: {pct(transfer['transported']['false_alarm_rate'])}; local calibration holds from "
              f"{transfer['smallest_calibration_that_holds']['local']}")
        allocation = m("wp10_allocation")
        key = "optimised (same ceiling) - design shares"
        sel = allocation["selection_on_one_half_valued_on_the_other_at_design_budget"]
        design = allocation["at_design_budget_base_prevalence"]["design shares"]
        print(f"  per 1000 screened at the assumed priors: {design['true_referrals_per_1000']:.2f} true referrals, up to "
              f"{design['false_referrals_per_1000_ceiling']:.0f} false referrals (ceiling)")
        print(f"  re-splitting the same ceiling: {sel[key + ' mean per 1000']:+.2f} true referrals per 1000 "
              f"[{sel[key + ' interval low']:+.2f}, {sel[key + ' interval high']:+.2f}] (chosen on half the people, valued on the other)")
    print("\n" + LINE)
    print(" Full tables and figures: reports/RESULTS.md")
    print(LINE)


if __name__ == "__main__":
    main()
