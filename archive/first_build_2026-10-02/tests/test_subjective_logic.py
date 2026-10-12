"""
Tests for Subjective Logic and Fusion Engine
"""

import pytest
import numpy as np
from src.fusion.subjective_logic import Opinion, subjective_logic_or, fuse_opinions_or
from src.fusion.decision import ClinicalDecisionReferee, ClinicalAction, TriageThresholds


def test_opinion_normalization():
    op = Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2)
    assert abs((op.belief + op.disbelief + op.uncertainty) - 1.0) < 1e-5
    assert 0.0 <= op.expected_probability <= 1.0


def test_vacuous_opinion():
    op = Opinion.vacuous()
    assert op.belief == 0.0
    assert op.disbelief == 0.0
    assert op.uncertainty == 1.0
    assert abs(op.expected_probability - op.base_rate) < 1e-5


def test_subjective_logic_or_identity():
    """Vacuous opinion should act as an identity element for disbelief in disjunctive fusion."""
    op_active = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
    op_missing = Opinion.vacuous()

    fused = subjective_logic_or(op_active, op_missing)
    assert np.isclose(fused.belief, op_active.belief, atol=1e-5)
    assert fused.disbelief <= op_active.disbelief


def test_subjective_logic_or_preserves_alarm():
    """If Structure expert sees clear CHD, missing or normal other modalities don't erase it."""
    op_structure = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
    op_function_missing = Opinion.vacuous()
    op_rhythm_normal = Opinion(belief=0.05, disbelief=0.85, uncertainty=0.10)

    fused = fuse_opinions_or([op_structure, op_function_missing, op_rhythm_normal])
    assert fused.belief > 0.80  # Alarm is strictly preserved


def test_clinical_decision_referee():
    referee = ClinicalDecisionReferee()

    # Case 1: High belief -> REFER
    op_alarm = Opinion(belief=0.75, disbelief=0.15, uncertainty=0.10)
    res_alarm = referee.evaluate(
        op_alarm,
        raw_opinions={"structure": op_alarm, "function": None, "rhythm": None},
        active_modalities=["structure"]
    )
    assert res_alarm.action == ClinicalAction.REFER

    # Case 2: High uncertainty & missing tests -> ACQUIRE_MISSING_TEST
    op_uncertain = Opinion(belief=0.10, disbelief=0.20, uncertainty=0.70)
    res_uncertain = referee.evaluate(
        op_uncertain,
        raw_opinions={"structure": None, "function": op_uncertain, "rhythm": None},
        active_modalities=["function"]
    )
    assert res_uncertain.action == ClinicalAction.ACQUIRE_MISSING_TEST

    # Case 3: High disbelief -> ROUTINE_CARE
    op_healthy = Opinion(belief=0.02, disbelief=0.88, uncertainty=0.10)
    res_healthy = referee.evaluate(
        op_healthy,
        raw_opinions={"structure": op_healthy, "function": op_healthy, "rhythm": op_healthy},
        active_modalities=["structure", "function", "rhythm"]
    )
    assert res_healthy.action == ClinicalAction.ROUTINE_CARE
