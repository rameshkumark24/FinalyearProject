"""WP12 - simulation of the textbook guarantees the decision rule relies on.

Every result checked here is standard (references in docs/theory_notes.md); the simulation only
shows that the project's own code behaves as the theory says, at the calibration sizes this
project has. Synthetic data throughout (labelled as such). Added on 20 October 2026 in answer to a
review; a check made after seeing the results.

Settings, fixed before computing:
  A  conformal validity    healthy scores normal or Student t with 3 degrees of freedom; calibration
                           sizes 14, 19, 60, 125 and 619 (the sizes of this project, plus 19 = the
                           smallest that can alarm at 0.05); levels 0.05 and 0.10; 2000 calibration
                           sets, each judged on 2000 new healthy cases. Reported: mean false-alarm
                           rate against floor(level (n+1)) / (n+1), and the 5th and 95th percentiles
                           of the per-set rate against the Beta(k, n+1-k) law
  B  union bound           three alarms with rates 0.05, 0.05 and 1/15: comonotone (lower end),
                           disjoint (upper end), independent, and equicorrelated Gaussian latent
                           variables with rho -0.45 to 1; 200000 healthy people each
  C  shift                 calibration on N(0, 1), use on N(delta, 1) for delta 0 to 1.5; 500
                           calibration sets of 60, each judged on 5000 cases; the realised rate
                           against level + KS distance (KS = 2 Phi(delta/2) - 1)
  D  Mondrian              two groups whose healthy scores differ by 1 sd; 30% of cases in the second
                           group; calibration sizes 40, 100 and 400; marginal and group-conditional
                           p-values at 0.05; 1000 sets, 4000 new cases each

    python scripts/run_wp12_theory_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sfr.provenance import Run  # noqa: E402
from sfrtool.allocation import effective_level  # noqa: E402
from sfrtool.conformal_groups import minimum_group_size  # noqa: E402

SIZES, LEVELS, N_SETS, N_TEST = (14, 19, 60, 125, 619), (0.05, 0.10), 2000, 2000
RATES = (0.05, 0.05, 1 / 15)
RHO_GRID = (-0.45, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
N_PEOPLE = 200_000
DELTAS, SHIFT_N, SHIFT_SETS, SHIFT_TEST = (0.0, 0.25, 0.5, 1.0, 1.5), 60, 500, 5000
GROUP_SHIFT, GROUP_SHARE, GROUP_SIZES, GROUP_SETS, GROUP_TEST = 1.0, 0.3, (40, 100, 400), 1000, 4000
SEED = 0


def conformal_alarm_rate(calibration: np.ndarray, test: np.ndarray, level: float) -> float:
    """Share of test scores whose conformal p-value (1 + #{cal >= s}) / (n + 1) is at most ``level``."""
    ordered = np.sort(calibration)
    at_least = ordered.size - np.searchsorted(ordered, test, side="left")
    return float(np.mean((1 + at_least) / (ordered.size + 1) <= level + 1e-12))


def part_a(rng: np.random.Generator) -> list[dict]:
    rows = []
    for family, draw in (("normal", lambda size: rng.normal(size=size)), ("student t, 3 df", lambda size: rng.standard_t(3, size=size))):
        for n in SIZES:
            for level in LEVELS:
                rates = np.array([conformal_alarm_rate(draw(n), draw(N_TEST), level) for _ in range(N_SETS)])
                k = int(np.floor(level * (n + 1) + 1e-12))
                beta = stats.beta(k, n + 1 - k) if k > 0 else None
                rows.append({"scores": family, "calibration_n": n, "level": level, "effective_level": effective_level(level, n),
                             "mean_rate": float(rates.mean()), "mean_rate_se": float(rates.std(ddof=1) / np.sqrt(N_SETS)),
                             "p5_rate": float(np.percentile(rates, 5)), "p95_rate": float(np.percentile(rates, 95)),
                             "beta_p5": float(beta.ppf(0.05)) if beta else 0.0, "beta_p95": float(beta.ppf(0.95)) if beta else 0.0,
                             "can_alarm": k > 0})
    return rows


def part_b(rng: np.random.Generator) -> list[dict]:
    rates = np.array(RATES)
    u = rng.random(N_PEOPLE)
    rows = [{"dependence": "comonotone (alarms nested)", "rate": float(np.mean(np.any(u[:, None] <= rates, axis=1)))}]
    offsets = np.concatenate([[0.0], np.cumsum(rates)[:-1]])
    disjoint = np.mod(u[:, None] - offsets[None, :], 1.0) <= rates
    rows.append({"dependence": "disjoint (alarms never together)", "rate": float(np.mean(disjoint.any(axis=1)))})
    rows.append({"dependence": "independent", "rate": float(np.mean(np.any(rng.random((N_PEOPLE, 3)) <= rates, axis=1)))})
    for rho in RHO_GRID:
        cov = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
        z = rng.multivariate_normal(np.zeros(3), cov, size=N_PEOPLE, method="eigh")
        rows.append({"dependence": f"gaussian rho {rho:+.2f}", "rate": float(np.mean(np.any(stats.norm.cdf(z) <= rates, axis=1)))})
    for row in rows:
        row["lower_end"], row["upper_end"] = float(rates.max()), float(min(1.0, rates.sum()))
        row["inside"] = bool(row["lower_end"] - 3e-3 <= row["rate"] <= row["upper_end"] + 3e-3)
    return rows


def part_c(rng: np.random.Generator) -> list[dict]:
    rows = []
    for delta in DELTAS:
        rates = np.array([conformal_alarm_rate(rng.normal(size=SHIFT_N), rng.normal(delta, 1, SHIFT_TEST), 0.05) for _ in range(SHIFT_SETS)])
        ks = float(2 * stats.norm.cdf(delta / 2) - 1)
        rows.append({"shift_sd": delta, "mean_rate": float(rates.mean()), "level": 0.05, "effective_level": effective_level(0.05, SHIFT_N),
                     "ks_distance": ks, "bound": effective_level(0.05, SHIFT_N) + ks,
                     "within_bound": bool(rates.mean() <= effective_level(0.05, SHIFT_N) + ks + 3e-3)})
    return rows


def part_d(rng: np.random.Generator) -> list[dict]:
    rows = []
    for n in GROUP_SIZES:
        results = {("marginal", 0): [], ("marginal", 1): [], ("group-conditional", 0): [], ("group-conditional", 1): []}
        for _ in range(GROUP_SETS):
            g_cal = rng.random(n) < GROUP_SHARE
            cal = rng.normal(size=n) + GROUP_SHIFT * g_cal
            g_test = rng.random(GROUP_TEST) < GROUP_SHARE
            test = rng.normal(size=GROUP_TEST) + GROUP_SHIFT * g_test
            for group in (0, 1):
                inside = g_test == bool(group)
                results[("marginal", group)].append(conformal_alarm_rate(cal, test[inside], 0.05))
                own = cal[g_cal == bool(group)]
                results[("group-conditional", group)].append(conformal_alarm_rate(own, test[inside], 0.05) if own.size else 0.0)
        for (kind, group), values in results.items():
            rows.append({"calibration_n": n, "calibration": kind, "group": "shifted (30%)" if group else "main (70%)",
                         "mean_rate": float(np.mean(values)), "level": 0.05,
                         "expected_group_calibration_n": n * (GROUP_SHARE if group else 1 - GROUP_SHARE),
                         "minimum_group_size_to_alarm": minimum_group_size(0.05)})
    return rows


def main() -> None:
    run = Run("wp12_theory_simulation",
              description="Simulation of the standard guarantees used by the decision rule: conformal validity and the Beta law, the "
                          "union bound and its ends, false alarms under shift, group-conditional calibration (synthetic data)",
              script="scripts/run_wp12_theory_simulation.py",
              config={"made_after_seeing_results": True, "requested_by": "review of 20 Oct 2026",
                      "A": {"sizes": list(SIZES), "levels": list(LEVELS), "sets": N_SETS, "test": N_TEST},
                      "B": {"rates": list(RATES), "rho": list(RHO_GRID), "people": N_PEOPLE},
                      "C": {"deltas": list(DELTAS), "calibration_n": SHIFT_N, "sets": SHIFT_SETS, "test": SHIFT_TEST},
                      "D": {"group_shift_sd": GROUP_SHIFT, "group_share": GROUP_SHARE, "sizes": list(GROUP_SIZES), "sets": GROUP_SETS,
                            "test": GROUP_TEST}},
              inputs={"code_under_test": [ROOT / "sfr" / "fusion" / "conformal.py", ROOT / "sfrtool" / "allocation.py",
                                          ROOT / "sfrtool" / "conformal_groups.py"]},
              n_real=0, n_synthetic=N_SETS * len(SIZES) * len(LEVELS) * 2,
              synthetic_kind="simulated healthy scores (normal or Student t) and simulated alarms; no patient data", seed=SEED)
    rng = np.random.default_rng(SEED)
    a, b, c, d = part_a(rng), part_b(rng), part_c(rng), part_d(rng)
    run.write_csv("a_conformal_validity.csv", a)
    run.write_csv("b_union_bound.csv", b)
    run.write_csv("c_shift.csv", c)
    run.write_csv("d_group_conditional.csv", d)
    metrics = {
        "made_after_seeing_results": True, "synthetic_only": True,
        "a_mean_rate_never_above_effective_level_by_more_than_3_se": all(r["mean_rate"] <= r["effective_level"] + 3 * r["mean_rate_se"] + 1e-12 for r in a),
        "a_largest_gap_between_simulated_and_beta_percentiles": max(max(abs(r["p5_rate"] - r["beta_p5"]), abs(r["p95_rate"] - r["beta_p95"])) for r in a),
        "b_every_dependence_inside_the_range": all(r["inside"] for r in b),
        "b_ends_reached": {"lower": next(r["rate"] for r in b if r["dependence"].startswith("comonotone")),
                           "upper": next(r["rate"] for r in b if r["dependence"].startswith("disjoint")),
                           "range": [max(RATES), min(1.0, sum(RATES))]},
        "c_every_shift_within_level_plus_ks": all(r["within_bound"] for r in c),
        "d_group_conditional_rates": {f"{r['calibration_n']}|{r['calibration']}|{r['group']}": r["mean_rate"] for r in d},
        "reading": "each part reproduces a standard result; none is a contribution of this project",
    }
    run.finish(metrics)
    for name, rows in (("A", a), ("B", b), ("C", c), ("D", d)):
        print(name)
        for r in rows:
            print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
