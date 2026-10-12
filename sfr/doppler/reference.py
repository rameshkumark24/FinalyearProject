"""Healthy reference for the Doppler timing measures, and a conformal p-value against it.

All NInFEA fetuses are healthy, so the function expert can learn what is normal and control how
often a healthy fetus is flagged. It cannot learn what is abnormal: no sensitivity is measured.

Each interval is compared with the value expected at the same cycle length (heart rate), and the
cycle length with the value expected at the same gestational age. The largest standardised
deviation is the score of a recording. Its p-value is its rank among the scores of healthy
recordings from other women (leave-one-woman-out), so a recording is never judged against a
reference that contains its own mother.
"""

from __future__ import annotations

import numpy as np

# measure -> the covariate its expected value depends on
REFERENCE_MEASURES = {
    "cycle_length_ms": "gestation_days",
    "a_to_v_peak_ms": "cycle_length_ms",
    "v_peak_to_closure_ms": "cycle_length_ms",
    "ejection_half_width_ms": "cycle_length_ms",
}


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line and the residual standard deviation."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 5:
        raise ValueError("Too few recordings for a reference")
    slope, intercept = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (0.0, float(np.mean(y)))
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.std(residual, ddof=2))


def fit_reference(rows: list[dict]) -> dict:
    """Reference lines from healthy recordings (dicts holding the measures and ``gestation_days``)."""
    reference = {}
    for measure, covariate in REFERENCE_MEASURES.items():
        x = np.array([r[covariate] for r in rows], dtype=float)
        y = np.array([r[measure] for r in rows], dtype=float)
        slope, intercept, sd = fit_line(x, y)
        reference[measure] = {"covariate": covariate, "slope": slope, "intercept": intercept, "sd": sd,
                              "n": int(np.sum(np.isfinite(x) & np.isfinite(y)))}
    return reference


def z_scores(row: dict, reference: dict) -> dict[str, float]:
    out = {}
    for measure, line in reference.items():
        value, x = row.get(measure, float("nan")), row.get(line["covariate"], float("nan"))
        out[measure] = (float((value - (line["slope"] * x + line["intercept"])) / line["sd"])
                        if np.isfinite(value) and np.isfinite(x) and line["sd"] > 0 else float("nan"))
    return out


def nonconformity(z: dict[str, float]) -> float:
    """Largest absolute standardised deviation. NaN if no measure could be read."""
    values = np.array([abs(v) for v in z.values() if np.isfinite(v)])
    return float(values.max()) if values.size else float("nan")


def conformal_p(score: float, calibration: np.ndarray) -> float:
    """Share of healthy calibration scores at least as extreme, counting the case itself."""
    calibration = np.asarray(calibration, dtype=float)
    calibration = calibration[np.isfinite(calibration)]
    if not np.isfinite(score):
        return float("nan")
    return float((1 + np.sum(calibration >= score)) / (calibration.size + 1))


def leave_one_woman_out(rows: list[dict]) -> list[dict]:
    """Score every recording against a reference built without any recording of the same woman.

    Returns one dict per recording: z-scores, score and conformal p-value. The calibration
    scores of a recording are the leave-one-woman-out scores of the other women's recordings.
    """
    women = sorted({r["woman"] for r in rows})
    scored = []
    for woman in women:
        reference = fit_reference([r for r in rows if r["woman"] != woman])
        for r in rows:
            if r["woman"] == woman:
                z = z_scores(r, reference)
                scored.append({"record": r["record"], "woman": woman, "z": z, "score": nonconformity(z)})
    scores = np.array([s["score"] for s in scored])
    owner = np.array([s["woman"] for s in scored])
    for s in scored:
        s["p_value"] = conformal_p(s["score"], scores[owner != s["woman"]])
    return scored


def intraclass_correlation(groups: list[np.ndarray]) -> float:
    """One-way random-effects ICC(1,1) for repeated measurements in groups of unequal size."""
    groups = [np.asarray(g, dtype=float)[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size >= 2]
    if len(groups) < 3:
        return float("nan")
    n_total = sum(g.size for g in groups)
    grand = np.concatenate(groups).mean()
    between = sum(g.size * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_total - len(groups))
    k0 = (n_total - sum(g.size ** 2 for g in groups) / n_total) / (len(groups) - 1)
    denominator = between + (k0 - 1) * within
    return float((between - within) / denominator) if denominator > 0 else float("nan")
