"""
SFR Framework - Model Accuracy & Performance Benchmark Reporter
===============================================================
Extracts and prints the terminal output of each model's accuracy,
AUROC, sensitivity, specificity, and evaluation metrics across
all modules of the multi-modal fetal cardiac screening system.
"""

import sys
import json
from pathlib import Path

# Force UTF-8 stdout if possible on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def print_banner(title: str):
    width = 85
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def report_structure_expert():
    print_banner("1. Structure Expert Models (Echocardiography Trimester Shift Resistance)")
    path = RESULTS_DIR / "structure_expert_trimester_benchmark.json"
    if not path.exists():
        print(f"[-] Missing {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trimester_data = data.get("trimester_domain_shift", {})
    header = f"{'Model Backbone':<36} | {'Within-Trimester':<17} | {'Cross-Trimester':<16} | {'Shift Drop (Gap)':<16}"
    print(header)
    print("-" * 90)
    for model, metrics in trimester_data.items():
        w_auc = metrics["within_trimester_auroc"]
        c_auc = metrics["cross_trimester_auroc"]
        gap = metrics["generalization_gap"]
        print(f"{model:<36} | {w_auc * 100:>6.2f}% (AUROC)  | {c_auc * 100:>6.2f}% (AUROC) | -{gap * 100:>5.2f}%")

    print("\n  Multi-View Dropout Robustness (FetalCLIP + LoRA):")
    print(f"  {'View Configuration':<34} | {'Views':<22} | {'AUROC':<10} | {'Sensitivity':<12}")
    print("  " + "-" * 82)
    view_data = data.get("view_dropout_robustness", {})
    for cfg, m in view_data.items():
        v_str = ", ".join(m["views"])
        print(f"  {cfg:<34} | {v_str:<22} | {m['auroc'] * 100:>6.2f}%   | {m['sensitivity'] * 100:>6.2f}%")


def report_rhythm_expert():
    print_banner("2. Rhythm Expert Models on Real Clinical Cohort (NIFEADB, N=26, 5-Fold Subject-Level CV)")
    path = RESULTS_DIR / "nifeadb_real_cohort_benchmark.json"
    if not path.exists():
        print(f"[-] Missing {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    header = f"{'Model Architecture':<42} | {'AUROC':<10} | {'Sens @ 90% Spec':<18} | {'Brier Score':<12}"
    print(header)
    print("-" * 88)
    for model, metrics in data.items():
        auc = metrics["mean_auroc"]
        sens = metrics["mean_sens_at_spec90"]
        brier = metrics["mean_brier_score"]
        print(f"{model:<42} | {auc * 100:>6.2f}%   | {sens * 100:>6.2f}%            | {brier:>8.4f}")


def report_leakage_gap():
    print_banner("3. Rhythm Expert Data Leakage Benchmark (Run R4)")
    path = RESULTS_DIR / "r4_leakage_gap_results.json"
    if not path.exists():
        print(f"[-] Missing {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seg = data["segment_split_metrics"]
    sub = data["subject_split_metrics"]

    header = f"{'Evaluation Protocol':<36} | {'Accuracy':<10} | {'AUROC':<10} | {'Sensitivity':<12} | {'Specificity':<12}"
    print(header)
    print("-" * 88)
    
    seg_acc = (seg['tp'] + seg['tn']) / (seg['tp'] + seg['tn'] + seg['fp'] + seg['fn']) * 100
    sub_acc = (sub['tp'] + sub['tn']) / (sub['tp'] + sub['tn'] + sub['fp'] + sub['fn']) * 100
    
    print(f"{'Segment-Level Split (Flawed / Leaked)':<36} | {seg_acc:>6.2f}%   | {seg['auroc'] * 100:>6.2f}%   | {seg['sensitivity'] * 100:>6.2f}%     | {seg['specificity'] * 100:>6.2f}%")
    print(f"{'Subject-Level Split (Honest / Clinical)':<36} | {sub_acc:>6.2f}%   | {sub['auroc'] * 100:>6.2f}%   | {sub['sensitivity'] * 100:>6.2f}%     | {sub['specificity'] * 100:>6.2f}%")
    print(f"\n  [!] Generalization Leakage Gap: {data['leakage_auroc_gap']:.4f} AUROC")


def report_fusion_models():
    print_banner("4. Multi-Modal Fusion Models Across Missing Modality Combinations")
    path = RESULTS_DIR / "fusion_benchmark_results.json"
    if not path.exists():
        print(f"[-] Missing {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = ["Subjective_Logic_OR", "Max_Rule", "Noisy_OR", "Dempster_Shafer", "Mean_Rule"]
    rule_labels = ["SL-OR (Ours)", "Max Rule", "Noisy-OR", "Dempster-Shafer", "Mean Rule"]

    pattern_display = {
        "All_Three_(S,F,R)": "All Three (Echo + Doppler + ECG)",
        "Echo+Doppler_(S,F,_)": "Echo + Doppler (ECG Missing)",
        "Echo+ECG_(S,_,R)": "Echo + ECG (Doppler Missing)",
        "Doppler+ECG_(_,F,R)": "Doppler + ECG (Echo Missing)",
        "Echo_Only_(S,_,_)": "Echo Only",
        "Doppler_Only_(_,F,_)": "Doppler Only",
        "ECG_Only_(_,_,R)": "ECG Only"
    }

    for p_key, p_name in pattern_display.items():
        if p_key not in data:
            continue
        p_data = data[p_key]
        print(f"\n  --- Modality Availability: {p_name} ---")
        header = f"  {'Fusion Operator':<24} | {'AUROC':<9} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'ECE':<8}"
        print(header)
        print("  " + "-" * 84)
        for r_key, r_label in zip(rules, rule_labels):
            m = p_data[r_key]
            sens = m["sensitivity"]
            spec = m["specificity"]
            # Population prevalence is 0.009 (18 positives / 2000 total)
            prev = 0.009
            acc = (sens * prev + spec * (1 - prev)) * 100
            print(f"  {r_label:<24} | {m['auroc'] * 100:>6.2f}% | {acc:>6.2f}%    | {sens * 100:>6.2f}%     | {spec * 100:>6.2f}%     | {m['ece']:>6.4f}")


def report_fetal_ecg_extraction():
    print_banner("5. Fetal ECG Extraction & QRS Detection Accuracy")
    p1 = RESULTS_DIR / "r1_r2_real_cinc2013_results.json"
    p2 = RESULTS_DIR / "cinc2013_full_cohort_benchmark.json"

    if p1.exists():
        with open(p1, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        print("  Benchmark on CinC 2013 Record a01:")
        print(f"  {'Method':<32} | {'F1-Score':<10} | {'Sensitivity (Recall)':<22} | {'PPV (Precision)':<15}")
        print("  " + "-" * 84)
        ts = d1["template_subtraction_metrics"]
        fica = d1["fastica_metrics"]
        print(f"  {'Template Subtraction (R1)':<32} | {ts['f1_score'] * 100:>6.2f}%   | {ts['sensitivity'] * 100:>6.2f}%                | {ts['ppv'] * 100:>6.2f}%")
        print(f"  {'FastICA (R2)':<32} | {fica['f1_score'] * 100:>6.2f}%   | {fica['sensitivity'] * 100:>6.2f}%                | {fica['ppv'] * 100:>6.2f}%")

    if p2.exists():
        with open(p2, "r", encoding="utf-8") as f:
            d2 = json.load(f)
        print(f"\n  Full Cohort Benchmark (PhysioNet CinC 2013 Set A, N={d2['total_records_processed']} Clinical Recordings):")
        print(f"  - Mean Beat F1-Score:    {d2['mean_f1_score'] * 100:.2f}% (Median: {d2['median_f1_score'] * 100:.2f}%)")
        print(f"  - Mean Sensitivity:      {d2['mean_sensitivity'] * 100:.2f}% (Median: {d2['median_sensitivity'] * 100:.2f}%)")
        print(f"  - Mean PPV (Precision):  {d2['mean_ppv'] * 100:.2f}% (Median: {d2['median_ppv'] * 100:.2f}%)")
        print(f"  - Mean Heart Rate Error: {d2['mean_hr_error_bpm']:.2f} bpm (Median: {d2['median_hr_error_bpm']:.2f} bpm)")


def report_multimodal_trust():
    print_banner("6. Doppler & ECG Multi-Modal Trust Check (NInFEA Cohort)")
    p = RESULTS_DIR / "ninfea_real_doppler_trust_benchmark.json"
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"  - Evaluated Real Multimodal Subjects: {d['subjects_evaluated']}")
    print(f"  - Cross-Modal Verification Pass Rate: {d['trust_verification_pass_rate'] * 100:.1f}%")
    print(f"  - Mean Multimodal Trust Score:        {d['mean_trust_score']:.3f}")
    print(f"  - False Alarm Rejection Belief:       {d['mean_false_alarm_belief'] * 100:.1f}%")


def main():
    print("\n" + "#" * 85)
    print("   PRENATAL MULTI-MODAL HEART SCREENING (SFR FRAMEWORK) - MODEL ACCURACY AUDIT")
    print("#" * 85)
    report_structure_expert()
    report_rhythm_expert()
    report_leakage_gap()
    report_fusion_models()
    report_fetal_ecg_extraction()
    report_multimodal_trust()
    print("\n" + "=" * 85)
    print("  [OK] All model accuracy metrics reported successfully.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
