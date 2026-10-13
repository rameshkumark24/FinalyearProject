"""WP5b - how much of the structure expert's result is more than its controls?

Added on 12 Oct 2026 AFTER the first results of run wp5_structure_<name> were seen: the probe on
images that hold no cleaned scan ("overlays only") scored almost as high as the probe on the
cleaned scan. This run puts numbers on that, from the scores and features of the structure run:

A. Paired differences in AUROC between the structure expert and each control, with a bootstrap
   interval over cases (the same cases are resampled for both scores).
B. The overlay-only probe inside the most common image size (the structure run had left it out).
C. What the overlay-only images are made of: how often the structure expert and the overlay-only
   probe alarm on the same cases.

Nothing here changes the structure run. It is a check made after seeing a result and is labelled so.

    python scripts/run_wp5b_structure_controls.py --structure-run wp5_structure_delta
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch  # noqa: F401  (must be imported before NumPy/SciPy code runs on this machine)
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sfr.eval.stats import auroc  # noqa: E402
from sfr.paths import RUNS  # noqa: E402
from sfr.provenance import Run, load_run  # noqa: E402

N_BOOT, SEED = 2000, 0
CONTROLS = ("fetalclip_overlay_only", "random_weights_scan", "no_anatomy_numbers", "fetalclip_unaltered_image")


def structure_module():
    spec = importlib.util.spec_from_file_location("run_wp5_structure", ROOT / "scripts" / "run_wp5_structure.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_wp5_structure"] = module
    spec.loader.exec_module(module)
    return module


def paired_difference(labels: np.ndarray, first: np.ndarray, second: np.ndarray, rng) -> dict:
    """AUROC of ``first`` minus AUROC of ``second`` on the same cases, with a bootstrap interval over cases."""
    point = auroc(labels, first) - auroc(labels, second)
    draws = []
    for _ in range(N_BOOT):
        index = rng.integers(0, labels.size, labels.size)
        if labels[index].min() == labels[index].max():
            continue
        draws.append(auroc(labels[index], first[index]) - auroc(labels[index], second[index]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(point), "ci": [float(low), float(high)], "interval_excludes_zero": bool(low > 0 or high < 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--structure-run", required=True)
    args = parser.parse_args()
    metrics, _ = load_run(args.structure_run)
    with open(RUNS / args.structure_run / "per_case.csv", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    labels = np.array([int(row["label"]) for row in cases])
    score = {name: np.array([float(row[f"score_{name}"]) for row in cases]) for name in ("fetalclip_scan", *CONTROLS)}
    rng = np.random.default_rng(SEED)

    part_a = {name: {"auroc_structure_expert": float(auroc(labels, score["fetalclip_scan"])), "auroc_control": float(auroc(labels, score[name])),
                     **paired_difference(labels, score["fetalclip_scan"], score[name], rng)} for name in CONTROLS}

    structure = structure_module()
    stored = np.load(RUNS / args.structure_run / "embeddings.npz", allow_pickle=False)
    size = np.array([row["image_size"] for row in cases])
    common = metrics["same_image_size"]["size"]
    member = size == common
    inside, _ = structure.evaluate(stored["fetalclip_overlay_only"][member], labels[member], with_permutation=False)
    part_b = {"size": common, "cases": int(member.sum()), "abnormal": int(labels[member].sum()), "overlay_only": inside,
              "structure_expert_same_cases": metrics["same_image_size"]["probes"]["fetalclip_scan"]}

    p_overlay = structure.conformal_p_values(stored["fetalclip_overlay_only"], labels)
    alarm_expert = np.array([row["alarm"] == "True" for row in cases])
    alarm_overlay = p_overlay <= structure.ALPHA
    abnormal = labels == 1
    part_c = {"alpha": structure.ALPHA,
              "abnormal_cases": int(abnormal.sum()),
              "caught_by_structure_expert": int(alarm_expert[abnormal].sum()),
              "caught_by_overlay_only_probe": int(alarm_overlay[abnormal].sum()),
              "caught_by_both": int((alarm_expert & alarm_overlay)[abnormal].sum()),
              "caught_by_structure_expert_only": int((alarm_expert & ~alarm_overlay)[abnormal].sum()),
              "normal_flagged_by_overlay_only_probe": int(alarm_overlay[~abnormal].sum()), "normal_cases": int((~abnormal).sum())}

    # D. the six numbers without anatomy (case averages stored by the structure run), by label
    numbers = stored["no_anatomy_numbers"]
    names = ("width", "height", "mean_grey", "scan_share", "coloured_removed", "white_strokes_removed")
    part_d = {name: {"median_normal": float(np.median(numbers[labels == 0, k])), "median_abnormal": float(np.median(numbers[labels == 1, k])),
                     "auroc_alone": float(auroc(labels, numbers[:, k]))} for k, name in enumerate(names)}
    planes = {entry["plane"]: entry for entry in metrics["by_plane"] if "auroc" in entry}

    run = Run(
        f"wp5b_structure_controls_{metrics['dataset']['name']}",
        description="Checks made after seeing the structure run: paired AUROC differences against the controls, the overlay-only probe inside one image size, and the overlap of alarms",
        script="scripts/run_wp5b_structure_controls.py",
        config={"structure_run": args.structure_run, "bootstrap": N_BOOT, "made_after_seeing_results": True},
        inputs={"scores": [RUNS / args.structure_run / "per_case.csv"], "features": [RUNS / args.structure_run / "embeddings.npz"]},
        n_real=int(labels.size),
        seed=SEED,
    )
    result = {"made_after_seeing_results": True, "A_paired_auroc_differences": part_a, "B_overlay_only_inside_one_image_size": part_b,
              "C_alarms_of_expert_and_overlay_probe": part_c,
              "D_numbers_without_anatomy_by_label": part_d,
              "E_plane_without_the_heart": {"abdomen": planes.get("Abdomen"), "four_chamber": planes.get("4CH"),
                                            "note": "the abdominal plane shows the stomach, spine and great vessels below the heart, not the heart itself"}}
    run.finish(result)
    for name, entry in part_a.items():
        print(f"{name:28s} expert {entry['auroc_structure_expert']:.3f} control {entry['auroc_control']:.3f} "
              f"difference {entry['difference']:+.3f} [{entry['ci'][0]:+.3f}, {entry['ci'][1]:+.3f}]")
    print("B:", part_b)
    print("C:", part_c)
    for name, entry in part_d.items():
        print(f"D: {name:22s} normal {entry['median_normal']:9.2f} abnormal {entry['median_abnormal']:9.2f} AUROC alone {entry['auroc_alone']:.3f}")


if __name__ == "__main__":
    main()
