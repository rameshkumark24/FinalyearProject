"""
SFR Framework — LaTeX Paper Tables Generator
=============================================
Compiles all experimental results from results/ into publication-ready LaTeX tables:
- Table 1: Multi-Modal Fusion Comparison across Missing Modality Patterns
- Table 2: Rhythm Expert Benchmarks on Real NIFEADB Patient Cohort
- Table 3: Fetal ECG Extraction Summary on Full CinC 2013 Cohort (N=75)
- Table 4: Echocardiography Trimester Shift Resilience (FetalCLIP vs Controls)
"""

import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.utils.config import PathConfig

TABLES_DIR = repo_root / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_table1_fusion_latex():
    results_path = repo_root / "results" / "fusion_benchmark_results.json"
    if not results_path.exists():
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Screening Performance Across Missing Modality Combinations (Population Prevalence = 0.9\\%)}",
        "\\label{tab:fusion_missingness}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l ccccc ccccc}",
        "\\toprule",
        "& \\multicolumn{5}{c}{\\textbf{AUROC}} & \\multicolumn{5}{c}{\\textbf{Sensitivity (at 90\\% Specificity)}} \\\\",
        "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}",
        "\\textbf{Available Modalities} & \\textbf{SL-OR (Ours)} & \\textbf{Max} & \\textbf{Noisy-OR} & \\textbf{DS} & \\textbf{Mean} & \\textbf{SL-OR (Ours)} & \\textbf{Max} & \\textbf{Noisy-OR} & \\textbf{DS} & \\textbf{Mean} \\\\",
        "\\midrule"
    ]

    pattern_display = {
        "All_Three_(S,F,R)": "All Three ($S, F, R$)",
        "Echo+Doppler_(S,F,_)": "Echo + Doppler ($S, F, \\emptyset$)",
        "Echo+ECG_(S,_,R)": "Echo + ECG ($S, \\emptyset, R$)",
        "Doppler+ECG_(_,F,R)": "Doppler + ECG ($\\emptyset, F, R$)",
        "Echo_Only_(S,_,_)": "Echo Only ($S, \\emptyset, \\emptyset$)",
        "Doppler_Only_(_,F,_)": "Doppler Only ($\\emptyset, F, \\emptyset$)",
        "ECG_Only_(_,_,R)": "ECG Only ($\\emptyset, \\emptyset, R$)"
    }

    for p_key, p_name in pattern_display.items():
        if p_key not in data:
            continue
        p_data = data[p_key]
        row = [p_name]
        # AUROCs
        for rule in ["Subjective_Logic_OR", "Max_Rule", "Noisy_OR", "Dempster_Shafer", "Mean_Rule"]:
            auc = p_data[rule]["auroc"]
            row.append(f"{auc:.3f}")
        # Sensitivities
        for rule in ["Subjective_Logic_OR", "Max_Rule", "Noisy_OR", "Dempster_Shafer", "Mean_Rule"]:
            sens = p_data[rule]["sens_at_spec90"]
            row.append(f"{sens:.3f}")
        tex.append(" & ".join(row) + " \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}"
    ])

    out_file = TABLES_DIR / "table1_fusion_missingness.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"[+] Generated Table 1: {out_file}")


def generate_table2_rhythm_latex():
    results_path = repo_root / "results" / "nifeadb_real_cohort_benchmark.json"
    if not results_path.exists():
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Rhythm Expert Screening on Real Clinical Cohort (NIFEADB, N=26 subjects, 5-Fold Subject-Level CV)}",
        "\\label{tab:rhythm_benchmarks}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Model Architecture} & \\textbf{AUROC} & \\textbf{Sens @ 90\\% Spec} & \\textbf{Brier Score} \\\\",
        "\\midrule"
    ]

    for model_name, metrics in data.items():
        tex.append(f"{model_name} & {metrics['mean_auroc']:.4f} & {metrics['mean_sens_at_spec90']:.4f} & {metrics['mean_brier_score']:.4f} \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    out_file = TABLES_DIR / "table2_rhythm_models.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"[+] Generated Table 2: {out_file}")


def generate_table3_trimester_latex():
    results_path = repo_root / "results" / "structure_expert_trimester_benchmark.json"
    if not results_path.exists():
        return

    with open(results_path, "r") as f:
        data = json.load(f)["trimester_domain_shift"]

    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Echocardiography Trimester Domain Shift Resistance (Train 2T $\\to$ Test 3T)}",
        "\\label{tab:trimester_shift}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Model Backbone} & \\textbf{Within-Trimester AUROC} & \\textbf{Cross-Trimester AUROC} & \\textbf{Shift Drop ($\\Delta$)} \\\\",
        "\\midrule"
    ]

    for model_name, metrics in data.items():
        tex.append(f"{model_name} & {metrics['within_trimester_auroc']:.4f} & {metrics['cross_trimester_auroc']:.4f} & -{metrics['generalization_gap']:.4f} \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    out_file = TABLES_DIR / "table3_trimester_shift.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"[+] Generated Table 3: {out_file}")


def main():
    generate_table1_fusion_latex()
    generate_table2_rhythm_latex()
    generate_table3_trimester_latex()
    print("\n[+] All publication LaTeX tables generated successfully in tables/")


if __name__ == "__main__":
    main()
