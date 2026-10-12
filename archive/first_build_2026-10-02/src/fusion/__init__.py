"""
SFR Framework — Fusion Engine
"""

from .subjective_logic import Opinion, subjective_logic_or, fuse_opinions_or
from .fusion_rules import FusionBenchmark
from .decision import ClinicalAction, TriageThresholds, ScreeningResult, ClinicalDecisionReferee

__all__ = [
    "Opinion",
    "subjective_logic_or",
    "fuse_opinions_or",
    "FusionBenchmark",
    "ClinicalAction",
    "TriageThresholds",
    "ScreeningResult",
    "ClinicalDecisionReferee"
]
