"""WP10 - splitting the false-referral budget between the experts.

Added on 20 October 2026 in answer to a review; a check made after seeing the results. The
union bound behind the rule is standard. This run asks only how a fixed false-referral ceiling
(the budget: the sum of the effective alarm levels) is best split between the experts, and how
much that is worth compared with the design levels and an equal split. Nothing is fitted: the
conformal p-values already computed by the expert runs are reused (as in wp8).

Settings fixed before computing (also in the run's config):
  budgets       caps on the effective ceiling: 0.10, 0.15, 0.20, 0.25, 0.30
  allocations   design shares (structure 1/4, function 1/4, rhythm 1/2 of the budget; at 0.20 the
                design 0.05/0.05/0.10); equal split (1/3 each); optimised (function kept at 1/4,
                structure and rhythm share the rest, rhythm at multiples of 1/15); optimised with
                the same effective ceiling as the design shares (the like-for-like comparison)
  function      its level is never optimised: it has no abnormal cases, so its sensitivity, and the
                value of giving it more budget, cannot be measured. Its people are not counted
                among the true referrals (a lower bound wherever function abnormalities exist)
  prevalences   the decision rule's priors (structure 0.009; function 0.002 and rhythm 0.001 are
                assumptions without a cited source), with the rhythm prevalence multiplied by
                0.5, 1, 2, 5 and 10 as a sensitivity grid
  selection     200 random halves by person (structure: DELTA cases with a defect; rhythm: NIFEADB
                arrhythmia recordings), seed 0: levels chosen on one half, valued on the other
  net benefit   harm of one unnecessary referral relative to the benefit of one needed referral:
                0.005, 0.01, 0.02, 0.05, 0.10 (the right value is a clinical judgement)
  size check    200000 simulated healthy people, three tests, total 0.20: independent, equicorrelated
                rho +0.5 and -0.45, and the two-test worst case for Simes at total 0.20

    python scripts/run_wp10_allocation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfr.eval.stats import clopper_pearson  # noqa: E402
from sfr.fusion.conformal import leave_one_out_p_values, p_value  # noqa: E402
from sfr.fusion.decision import DecisionConfig  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402
from sfrtool.allocation import (Allocation, best_allocation, bonferroni_any, evaluate, gaussian_null_p, holm_any,  # noqa: E402
                                net_benefit, per_thousand, simes_any, simes_worst_case_two)

STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN, TRADEOFF_RUN = "wp5_structure_delta", "wp2b_function_reference", "wp4b_rhythm_intervals", "wp8_tradeoff"
RHYTHM_500HZ_RUN = "wp4e_rhythm_500hz"
RHYTHM_SCORE = "oof_logistic_rhythm"
EXPERTS = ("structure", "function", "rhythm")
DESIGN = dict(DecisionConfig().alpha)
PRIOR = dict(DecisionConfig().prior)
BUDGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
DESIGN_SHARES = {"structure": 1 / 4, "function": 1 / 4, "rhythm": 1 / 2}
RHYTHM_PREVALENCE_FACTORS = (0.5, 1.0, 2.0, 5.0, 10.0)
HARM_RATIOS = (0.005, 0.01, 0.02, 0.05, 0.10)
N_SPLITS, SEED = 200, 0
N_SIMULATED, SIZE_TOTAL = 200_000, 0.20
DEPENDENCE = {"independent": 0.0, "positive (rho +0.5)": 0.5, "negative (rho -0.45)": -0.45}
STEP = 0.0025


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rhythm_p(run_name: str) -> dict:
    rows = read_csv(RUNS / run_name / "per_record_scores.csv")
    label = np.array([int(r["label"]) for r in rows])
    score = np.array([float(r[RHYTHM_SCORE]) for r in rows])
    normal = score[label == 0]
    return {"healthy": leave_one_out_p_values(normal), "finding": np.array([p_value(s, normal) for s in score[label == 1]]),
            "n_calibration": int(normal.size)}


def p_values() -> dict:
    """Conformal p-values of every expert, as computed for the reported results (same as wp8)."""
    cases = read_csv(RUNS / STRUCTURE_RUN / "per_case.csv")
    label = np.array([int(r["label"]) for r in cases])
    structure = np.array([float(r["p_value"]) for r in cases])
    function = np.array([float(r["p_value"]) for r in read_csv(RUNS / FUNCTION_RUN / "per_recording.csv")])
    return {"structure": {"healthy": structure[label == 0], "finding": structure[label == 1], "n_calibration": None},
            "function": {"healthy": function, "finding": None, "n_calibration": None},
            "rhythm": rhythm_p(RHYTHM_RUN)}


def check_against_tradeoff(p: dict) -> None:
    """The sensitivities used here must equal those published by wp8 at its levels."""
    for row in read_csv(RUNS / TRADEOFF_RUN / "expert_curves.csv"):
        if row["sensitivity"] == "":
            continue
        level = float(row["alarm_level"])
        caught = int(np.sum(p[row["expert"]]["finding"] <= level + 1e-12))
        if caught != int(row["finding_caught"]):
            raise ValueError(f"wp8 curve differs for {row['expert']} at {level}: {caught} against {row['finding_caught']}")


def measured_range(p: dict, effective: dict) -> tuple[float, float]:
    """Lowest and highest healthy referral rate for any dependence, from the measured healthy alarm rates."""
    rates = [float(np.mean(p[k]["healthy"] <= effective[k] + 1e-12)) for k in EXPERTS if effective.get(k, 0) > 0]
    if not rates:
        return 0.0, 0.0
    return max(rates), min(1.0, sum(rates))


def allocations(budget: float, p: dict, prevalence: dict) -> list[Allocation]:
    finding = {k: p[k]["finding"] for k in EXPERTS}
    n_cal = {k: p[k]["n_calibration"] for k in EXPERTS}
    design = evaluate("design shares", {k: budget * DESIGN_SHARES[k] for k in EXPERTS}, finding, prevalence, n_cal)
    equal = evaluate("equal split", {k: budget / 3 for k in EXPERTS}, finding, prevalence, n_cal)
    fixed = {"function": budget * DESIGN_SHARES["function"]}
    optimised = best_allocation(budget, finding, prevalence, n_cal, fixed, ("rhythm", "structure"), STEP)
    same = best_allocation(design.effective_ceiling, finding, prevalence, n_cal, fixed, ("rhythm", "structure"), STEP)
    same = Allocation("optimised, same ceiling as design shares", same.levels, same.effective, same.sensitivity, same.value)
    return [design, equal, optimised, same]


def row_of(alloc: Allocation, budget: float, factor: float, prevalence: dict, p: dict, source: str) -> dict:
    low, high = measured_range(p, alloc.effective)
    healthy_share = 1 - sum(prevalence.values())
    return {"rhythm_source": source, "budget": budget, "rhythm_prevalence_factor": factor, "allocation": alloc.kind,
            **{f"{k}_level": alloc.levels[k] for k in EXPERTS}, **{f"{k}_effective": alloc.effective[k] for k in EXPERTS},
            "effective_ceiling": alloc.effective_ceiling, "unspent": alloc.ceiling - alloc.effective_ceiling,
            "structure_sensitivity": alloc.sensitivity["structure"], "rhythm_sensitivity": alloc.sensitivity["rhythm"],
            "true_referrals_per_1000": per_thousand(alloc.value),
            "false_referrals_per_1000_ceiling": per_thousand(healthy_share * alloc.effective_ceiling),
            "false_referrals_per_1000_measured_low": per_thousand(healthy_share * low),
            "false_referrals_per_1000_measured_high": per_thousand(healthy_share * high)}


def split_half(p: dict, budget: float, prevalence: dict, rng: np.random.Generator) -> dict:
    """Levels chosen on half of the people with each finding, valued on the other half."""
    n_cal = {k: p[k]["n_calibration"] for k in EXPERTS}
    fixed = {"function": budget * DESIGN_SHARES["function"]}
    design_levels = {k: budget * DESIGN_SHARES[k] for k in EXPERTS}
    equal_levels = {k: budget / 3 for k in EXPERTS}
    gains = {"optimised - design shares": [], "optimised (same ceiling) - design shares": [], "optimised - equal split": []}
    changed = 0
    for _ in range(N_SPLITS):
        halves = {}
        for k in ("structure", "rhythm"):
            order = rng.permutation(p[k]["finding"].size)
            halves[k] = (p[k]["finding"][order[: order.size // 2]], p[k]["finding"][order[order.size // 2:]])
        choose = {"structure": halves["structure"][0], "function": None, "rhythm": halves["rhythm"][0]}
        value = {"structure": halves["structure"][1], "function": None, "rhythm": halves["rhythm"][1]}
        design = evaluate("design", design_levels, value, prevalence, n_cal)
        chosen = best_allocation(budget, choose, prevalence, n_cal, fixed, ("rhythm", "structure"), STEP)
        chosen_same = best_allocation(design.effective_ceiling, choose, prevalence, n_cal, fixed, ("rhythm", "structure"), STEP)
        valued = evaluate("chosen", chosen.levels, value, prevalence, n_cal)
        valued_same = evaluate("chosen", chosen_same.levels, value, prevalence, n_cal)
        equal = evaluate("equal", equal_levels, value, prevalence, n_cal)
        gains["optimised - design shares"].append(per_thousand(valued.value - design.value))
        gains["optimised (same ceiling) - design shares"].append(per_thousand(valued_same.value - design.value))
        gains["optimised - equal split"].append(per_thousand(valued.value - equal.value))
        changed += int(not np.isclose(chosen_same.effective["rhythm"], design.effective["rhythm"]))
    out = {}
    for name, values in gains.items():
        values = np.array(values)
        out[name] = {"mean_per_1000": float(values.mean()), "interval_per_1000": [float(v) for v in np.percentile(values, [2.5, 97.5])],
                     "share_above_0": float(np.mean(values > 1e-12)), "share_below_0": float(np.mean(values < -1e-12))}
    out["share_of_splits_choosing_another_rhythm_level_at_the_same_ceiling"] = changed / N_SPLITS
    return out


def size_check(rng: np.random.Generator) -> list[dict]:
    rows = []
    levels = [SIZE_TOTAL / 3] * 3
    for name, rho in DEPENDENCE.items():
        p = rng.random((N_SIMULATED, 3)) if rho == 0 else gaussian_null_p(N_SIMULATED, 3, rho, rng)
        rows.append({"dependence": name, "tests": 3, "total": SIZE_TOTAL, "n_simulated": N_SIMULATED,
                     "bonferroni": float(bonferroni_any(p, levels).mean()), "holm": float(holm_any(p[:20_000], levels).mean()),
                     "holm_equals_bonferroni_on_first_20000": bool(np.array_equal(holm_any(p[:20_000], levels), bonferroni_any(p[:20_000], levels))),
                     "simes": float(simes_any(p, SIZE_TOTAL).mean()),
                     "simes_guaranteed": "yes (positive dependence)" if rho > 0 else ("yes (independence)" if rho == 0 else "no")})
    p = simes_worst_case_two(N_SIMULATED, SIZE_TOTAL, rng)
    rows.append({"dependence": "worst case for Simes (two tests)", "tests": 2, "total": SIZE_TOTAL, "n_simulated": N_SIMULATED,
                 "bonferroni": float(bonferroni_any(p, [SIZE_TOTAL / 2] * 2).mean()),
                 "holm": float(holm_any(p[:20_000], [SIZE_TOTAL / 2] * 2).mean()),
                 "holm_equals_bonferroni_on_first_20000": bool(np.array_equal(holm_any(p[:20_000], [SIZE_TOTAL / 2] * 2),
                                                                              bonferroni_any(p[:20_000], [SIZE_TOTAL / 2] * 2))),
                 "simes": float(simes_any(p, SIZE_TOTAL).mean()), "simes_guaranteed": "no"})
    return rows


def simes_illustration(p: dict, rng: np.random.Generator) -> dict:
    """Referral probability of a person with one finding when the two other tests behave like independent healthy tests.

    An assumption, made only for this illustration: the other two p-values are independent
    uniforms. Equal levels (total 0.20) for both rules so that they are comparable.
    """
    others = rng.random((20_000, 2))
    out = {}
    for k in ("structure", "rhythm"):
        own, bonf, simes = [], [], []
        for value in p[k]["finding"]:
            full = np.column_stack([np.full(others.shape[0], value), others])
            own.append(float(value <= SIZE_TOTAL / 3 + 1e-12))
            bonf.append(float(bonferroni_any(full, [SIZE_TOTAL / 3] * 3).mean()))
            simes.append(float(simes_any(full, SIZE_TOTAL).mean()))
        out[k] = {"own_test_only": float(np.mean(own)), "bonferroni_any_alarm": float(np.mean(bonf)), "simes": float(np.mean(simes)),
                  "n": int(p[k]["finding"].size),
                  "note": "the gap between own test and any-alarm is chance alarms of the two other tests, not detection"}
    return out


def decision_curve(p: dict, prevalence: dict) -> list[dict]:
    """Net benefit per 1000 at the design budget for the design, the like-for-like optimum, structure alone, refer all, refer none."""
    budget = sum(DESIGN.values())
    finding = {k: p[k]["finding"] for k in EXPERTS}
    n_cal = {k: p[k]["n_calibration"] for k in EXPERTS}
    design = evaluate("design (0.05, 0.05, 0.10)", DESIGN, finding, prevalence, n_cal)
    same = allocations(budget, p, prevalence)[3]
    no_rhythm = evaluate("without the rhythm expert", {"structure": DESIGN["structure"], "function": DESIGN["function"], "rhythm": 0.0},
                         finding, prevalence, n_cal)
    healthy = 1 - sum(prevalence.values())
    counted = prevalence["structure"] + prevalence["rhythm"]
    rows = []
    for harm in HARM_RATIOS:
        for alloc in (design, same, no_rhythm):
            low, high = measured_range(p, alloc.effective)
            rows.append({"harm_ratio": harm, "strategy": alloc.kind, "true_referrals_per_1000": per_thousand(alloc.value),
                         "false_referrals_per_1000_ceiling": per_thousand(healthy * alloc.effective_ceiling),
                         "net_benefit_per_1000_at_ceiling": per_thousand(net_benefit(alloc.value, healthy * alloc.effective_ceiling, harm)),
                         "net_benefit_per_1000_at_measured_low": per_thousand(net_benefit(alloc.value, healthy * low, harm))})
        rows.append({"harm_ratio": harm, "strategy": "refer everyone", "true_referrals_per_1000": per_thousand(counted),
                     "false_referrals_per_1000_ceiling": per_thousand(healthy),
                     "net_benefit_per_1000_at_ceiling": per_thousand(net_benefit(counted, healthy, harm)),
                     "net_benefit_per_1000_at_measured_low": per_thousand(net_benefit(counted, healthy, harm))})
        rows.append({"harm_ratio": harm, "strategy": "refer no one", "true_referrals_per_1000": 0.0, "false_referrals_per_1000_ceiling": 0.0,
                     "net_benefit_per_1000_at_ceiling": 0.0, "net_benefit_per_1000_at_measured_low": 0.0})
    return rows


def with_and_without_rhythm(p: dict, prevalence: dict) -> dict:
    finding = {k: p[k]["finding"] for k in EXPERTS}
    n_cal = {k: p[k]["n_calibration"] for k in EXPERTS}
    rows = {}
    for name, levels in (("three experts, design levels", DESIGN),
                         ("structure and function only", {"structure": DESIGN["structure"], "function": DESIGN["function"], "rhythm": 0.0}),
                         ("structure and function, rhythm's budget given to structure",
                          {"structure": DESIGN["structure"] + 1 / 15, "function": DESIGN["function"], "rhythm": 0.0})):
        alloc = evaluate(name, levels, finding, prevalence, n_cal)
        low, high = measured_range(p, alloc.effective)
        rows[name] = {"levels": alloc.levels, "nominal_ceiling": alloc.ceiling, "effective_ceiling": alloc.effective_ceiling,
                      "measured_healthy_range_any_dependence": [low, high], "sensitivity": alloc.sensitivity,
                      "true_referrals_per_1000": per_thousand(alloc.value),
                      "false_referrals_per_1000_ceiling": per_thousand((1 - sum(prevalence.values())) * alloc.effective_ceiling)}
    return rows


def main() -> None:
    for name in (STRUCTURE_RUN, FUNCTION_RUN, RHYTHM_RUN, TRADEOFF_RUN):
        load_run(name)
    p = p_values()
    check_against_tradeoff(p)
    sources = {"wp4b (the rhythm expert as used)": p}
    has_500 = (RUNS / RHYTHM_500HZ_RUN / "metrics.json").exists()
    if has_500:
        load_run(RHYTHM_500HZ_RUN)
        sources["wp4e (500 Hz check)"] = {**p, "rhythm": rhythm_p(RHYTHM_500HZ_RUN)}

    rng = np.random.default_rng(SEED)
    rows, selection = [], []
    for source, ps in sources.items():
        for budget in BUDGETS:
            for factor in RHYTHM_PREVALENCE_FACTORS:
                prevalence = {**PRIOR, "rhythm": PRIOR["rhythm"] * factor}
                for alloc in allocations(budget, ps, prevalence):
                    rows.append(row_of(alloc, budget, factor, prevalence, ps, source))
                if source.startswith("wp4b"):
                    result = split_half(ps, budget, prevalence, rng)
                    selection.append({"budget": budget, "rhythm_prevalence_factor": factor,
                                      **{f"{k} mean per 1000": v["mean_per_1000"] for k, v in result.items() if isinstance(v, dict)},
                                      **{f"{k} interval low": v["interval_per_1000"][0] for k, v in result.items() if isinstance(v, dict)},
                                      **{f"{k} interval high": v["interval_per_1000"][1] for k, v in result.items() if isinstance(v, dict)},
                                      **{f"{k} share above 0": v["share_above_0"] for k, v in result.items() if isinstance(v, dict)},
                                      "share choosing another rhythm level at the same ceiling":
                                          result["share_of_splits_choosing_another_rhythm_level_at_the_same_ceiling"]})
        print(f"{source}: allocations done", flush=True)
    sizes = size_check(np.random.default_rng(SEED))
    illustration = simes_illustration(p, np.random.default_rng(SEED))
    curve = decision_curve(p, PRIOR)
    headline = with_and_without_rhythm(p, PRIOR)

    design_rows = [r for r in rows if r["rhythm_source"].startswith("wp4b") and np.isclose(r["budget"], sum(DESIGN.values()))
                   and r["rhythm_prevalence_factor"] == 1.0]
    by_kind = {r["allocation"]: r for r in design_rows}
    at_design = next(s for s in selection if np.isclose(s["budget"], sum(DESIGN.values())) and s["rhythm_prevalence_factor"] == 1.0)
    largest_gain = max(abs(s["optimised (same ceiling) - design shares mean per 1000"]) for s in selection)
    metrics = {
        "made_after_seeing_results": True,
        "requested_by": "review of 20 Oct 2026",
        "design_levels": DESIGN, "priors": PRIOR,
        "priors_note": "structure 0.009 is the decision rule's prior; function 0.002 and rhythm 0.001 are assumptions without a cited source "
                       "[unverified - needs a source]; the rhythm prevalence is varied over a grid for that reason",
        "at_design_budget_base_prevalence": by_kind,
        "selection_on_one_half_valued_on_the_other_at_design_budget": at_design,
        "largest_mean_gain_same_ceiling_per_1000_over_grid": largest_gain,
        "simes_illustration_assuming_independent_healthy_other_tests": illustration,
        "size_check": sizes,
        "headline_with_and_without_rhythm": headline,
        "rhythm_500hz_variant_included": has_500,
        "function_note": "the function level is never optimised and its people are never counted: it has no abnormal cases",
        "reading": ("The union bound is standard. Splitting a fixed ceiling differently changes which condition is detected more often; "
                    "it never improves detection of any one condition beyond what that expert gives at its own level. Gains are "
                    "reported per 1000 people screened at assumed prevalences; with these sample sizes they are small and uncertain."),
    }
    run = Run("wp10_allocation",
              description="Budgeted split of the false-referral ceiling between the experts, compared with the design levels and an equal "
                          "split; Holm and Simes; net benefit per 1000. A check made after seeing the results.",
              script="scripts/run_wp10_allocation.py",
              config={"made_after_seeing_results": True, "budgets": list(BUDGETS), "budget_is": "sum of effective alarm levels",
                      "design_shares": DESIGN_SHARES, "function_level": "1/4 of the budget, never optimised (no abnormal cases)",
                      "rhythm_levels": "multiples of 1/15 (14 healthy calibration recordings)", "structure_grid_step": STEP,
                      "priors": PRIOR, "rhythm_prevalence_factors": list(RHYTHM_PREVALENCE_FACTORS),
                      "split_half": {"splits": N_SPLITS, "unit": "person with the finding", "seed": SEED},
                      "harm_ratios": list(HARM_RATIOS), "size_check": {"n": N_SIMULATED, "total": SIZE_TOTAL, "dependence": DEPENDENCE},
                      "tie_break": "same value: smaller effective ceiling first"},
              inputs={"structure": [RUNS / STRUCTURE_RUN / "per_case.csv"], "function": [RUNS / FUNCTION_RUN / "per_recording.csv"],
                      "rhythm": [RUNS / RHYTHM_RUN / "per_record_scores.csv"]
                      + ([RUNS / RHYTHM_500HZ_RUN / "per_record_scores.csv"] if has_500 else []),
                      "wp8_curves": [RUNS / TRADEOFF_RUN / "expert_curves.csv"]},
              n_real=int(p["structure"]["healthy"].size + p["structure"]["finding"].size + p["function"]["healthy"].size
                         + p["rhythm"]["healthy"].size + p["rhythm"]["finding"].size),
              seed=SEED)
    run.write_csv("allocations.csv", rows)
    run.write_csv("selection_split_half.csv", selection)
    run.write_csv("size_check.csv", sizes)
    run.write_csv("decision_curve.csv", curve)
    run.finish(metrics)

    print("\nAt the design budget, base prevalences (rhythm curve of wp4b):")
    for kind, r in by_kind.items():
        print(f"  {kind:42s} levels s {r['structure_level']:.4f} f {r['function_level']:.4f} r {r['rhythm_level']:.4f} "
              f"effective ceiling {r['effective_ceiling']:.4f}  true/1000 {r['true_referrals_per_1000']:.2f}  "
              f"false/1000 (ceiling) {r['false_referrals_per_1000_ceiling']:.1f}")
    print("Chosen on one half, valued on the other (design budget, base prevalences):")
    for key, value in at_design.items():
        print(f"  {key}: {value}")
    print("Size check:")
    for r in sizes:
        print(f"  {r['dependence']:34s} Bonferroni {r['bonferroni']:.4f}  Holm {r['holm']:.4f}  Simes {r['simes']:.4f}")
    print("Simes illustration:", illustration)


if __name__ == "__main__":
    main()
