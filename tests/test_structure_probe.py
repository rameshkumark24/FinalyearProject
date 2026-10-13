"""Tests of the structure expert's evaluation code (scripts/run_wp5_structure.py).

No labelled echo data is on disk, so the image part cannot be tested on real cases. These tests
check the statistics on SYNTHETIC feature vectors, made here and used nowhere else: a planted
difference must be found, pure noise must not be, cases must not be split, and the conformal
alarm must keep its false-alarm level. Nothing in this file is a result.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import torch  # noqa: F401  (before NumPy work on this machine)
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_wp5_structure", ROOT / "scripts" / "run_wp5_structure.py")
structure = importlib.util.module_from_spec(spec)
sys.modules["run_wp5_structure"] = structure
spec.loader.exec_module(structure)


def synthetic_cases(shift: float, n_cases: int = 300, seed: int = 1):
    """Synthetic feature vectors: noise, plus ``shift`` in three dimensions for the 'abnormal' cases."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_cases) < 0.2).astype(int)
    x = rng.normal(size=(n_cases, 40))
    x[y == 1, :3] += shift
    return x, y


def test_planted_difference_is_found_and_noise_is_not():
    x, y = synthetic_cases(shift=1.5)
    found, _ = structure.evaluate(x, y, with_permutation=False)
    assert found["auroc"] > 0.85
    x, y = synthetic_cases(shift=0.0)
    noise, _ = structure.evaluate(x, y, with_permutation=False)
    assert 0.35 < noise["auroc"] < 0.65


def test_permutation_p_value_separates_signal_from_noise():
    x, y = synthetic_cases(shift=1.5, n_cases=150)
    assert structure.permutation_p(x, y, n_perm=40) < 0.05
    x, y = synthetic_cases(shift=0.0, n_cases=150)
    assert structure.permutation_p(x, y, n_perm=40) > 0.05


def test_conformal_alarm_keeps_its_false_alarm_level():
    x, y = synthetic_cases(shift=1.5, n_cases=600, seed=3)
    p = structure.conformal_p_values(x, y)
    assert np.isfinite(p).all() and (p > 0).all() and (p <= 1).all()
    summary = structure.alarm_summary(y, p, [""] * len(y))
    # level 0.05; with about 480 synthetic normal cases the share stays well under 0.10
    assert summary["false_alarms"]["share"] < 0.10
    assert summary["sensitivity"]["share"] > 0.3


def test_images_of_one_case_are_averaged_and_never_split():
    rows = [{"case": "A", "label": "0", "subtype": "", "plane": "p1", "path": "a1"},
            {"case": "A", "label": "0", "subtype": "", "plane": "p2", "path": "a2"},
            {"case": "B", "label": "1", "subtype": "x", "plane": "p1", "path": "b1"}]
    features = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    cases, x, y, subtype = structure.by_case(rows, features)
    assert cases == ["A", "B"] and y.tolist() == [0, 1] and subtype == ["", "x"]
    assert np.allclose(x, [[2.0, 0.0], [0.0, 5.0]])
    only_p1 = np.array([row["plane"] == "p1" for row in rows])
    _, x_p1, _, _ = structure.by_case(rows, features, only_p1)
    assert np.allclose(x_p1, [[1.0, 0.0], [0.0, 5.0]])


def test_a_case_with_two_labels_is_refused():
    rows = [{"case": "A", "label": "0", "subtype": "", "plane": "", "path": "a1"},
            {"case": "A", "label": "1", "subtype": "", "plane": "", "path": "a2"}]
    with pytest.raises(ValueError):
        structure.by_case(rows, np.zeros((2, 2)))


def test_cleaning_in_one_pass_gives_the_same_images_as_the_package(tmp_path):
    from PIL import Image

    from sfr.echo.deidentify import deidentify, overlay_only

    # A synthetic frame: a textured grey disc (the "scan") on black, thin white strokes inside and outside it.
    rng = np.random.default_rng(5)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    rows, cols = np.mgrid[:300, :400]
    disc = (rows - 160) ** 2 + (cols - 200) ** 2 < 110 ** 2
    frame[disc] = rng.integers(40, 200, size=(int(disc.sum()), 1), dtype=np.uint8)
    frame[20:23, 30:150] = 255
    frame[150:152, 150:260] = 255
    path = tmp_path / "frame.png"
    Image.fromarray(frame).save(path)

    scan, removed, rgb, numbers = structure.clean_one(str(path))
    assert np.array_equal(rgb, frame)
    assert np.array_equal(scan, deidentify(frame).image)
    assert np.array_equal(removed, overlay_only(frame))
    assert numbers[:2] == [400, 300]


def test_index_with_a_bad_label_is_refused(tmp_path):
    index = tmp_path / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "label", "path"])
        writer.writerow(["A", "abnormal", "a.png"])
    with pytest.raises(ValueError):
        structure.read_index(index)
