"""The fast helpers of scripts/run_wp6b_fusion_structure.py give the same numbers as the originals.

Inputs are random numbers made here (synthetic, used for this comparison only).
"""

import importlib.util
import sys
from pathlib import Path

import torch  # noqa: F401  (before NumPy work on this machine)
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new, old = load("run_wp6b_fusion_structure"), load("run_wp6_fusion")


def test_dempster_for_all_cases_at_once_equals_the_case_by_case_rule():
    rng = np.random.default_rng(0)
    strengths = rng.random((400, 3))
    strengths[rng.random((400, 3)) < 0.25] = np.nan
    strengths[:5] = [[0.0, 1.0, 0.5], [1.0, 0.0, np.nan], [np.nan, np.nan, np.nan], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    assert np.allclose(new.dempster_columns(strengths), old.rules.dempster(strengths), equal_nan=True)


def test_fast_threshold_equals_the_original_also_with_ties():
    rng = np.random.default_rng(1)
    for scores in (rng.random(500), np.round(rng.random(500), 1), np.round(rng.random(60), 2), np.ones(40)):
        for target in (0.05, 0.10, 0.15, 0.5):
            assert new.threshold_for_rate(scores, target) == old.threshold_for_rate(scores, target)


def test_pairing_matches_the_original():
    rng = np.random.default_rng(2)
    first, second = rng.random(37), rng.random(11)
    for coupling in new.COUPLINGS:
        for mine, theirs in zip(new.pair(first, second, coupling), old.pair(first, second, coupling)):
            assert np.array_equal(mine, theirs)
