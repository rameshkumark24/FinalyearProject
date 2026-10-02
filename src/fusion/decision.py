"""
SFR Framework — Clinical Decision Triage
=========================================
Converts a fused Subjective Logic Opinion (b, d, u, a) into an actionable,
clinically safe tri-state decision:
1. REFER (urgent pediatric cardiology consultation)
2. ACQUIRE_MISSING_TEST (uncertainty too high for routine reassurance; request Doppler/Echo/ECG)
3. ROUTINE_CARE (all available tests reassuring; proceed with standard obstetrics follow-up)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .subjective_logic import Opinion


class ClinicalAction(str, Enum):
    REFER = "REFER"
    ACQUIRE_MISSING_TEST = "ACQUIRE_MISSING_TEST"
    ROUTINE_CARE = "ROUTINE_CARE"


@dataclass
class TriageThresholds:
    referral_belief_threshold: float = 0.40  # Belief in CHD/defect >= 0.4 -> Refer
    referral_prob_threshold: float = 0.35    # Or projected prob >= 0.35 -> Refer
    routine_disbelief_threshold: float = 0.70 # Strong evidence of normality needed
    max_acceptable_uncertainty: float = 0.50 # Uncertainty above 0.5 cannot be routine care


@dataclass
class ScreeningResult:
    action: ClinicalAction
    fused_opinion: Opinion
    explanation: str
    missing_modalities: list[str]
    raw_expert_opinions: Dict[str, Optional[Opinion]]


class ClinicalDecisionReferee:
    def __init__(self, thresholds: Optional[TriageThresholds] = None):
        self.thresholds = thresholds or TriageThresholds()

    def evaluate(
        self,
        fused_opinion: Opinion,
        raw_opinions: Dict[str, Optional[Opinion]],
        active_modalities: list[str]
    ) -> ScreeningResult:
        all_modalities = ["structure", "function", "rhythm"]
        missing = [m for m in all_modalities if m not in active_modalities or raw_opinions.get(m) is None]

        b = fused_opinion.belief
        d = fused_opinion.disbelief
        u = fused_opinion.uncertainty
        prob = fused_opinion.expected_probability

        # Triage Rule 1: High belief in abnormality -> REFER IMMEDIATELY
        if b >= self.thresholds.referral_belief_threshold or prob >= self.thresholds.referral_prob_threshold:
            action = ClinicalAction.REFER
            explanation = (
                f"Referral recommended: fused belief {b:.2f} or expected risk {prob:.2f} "
                f"exceeds referral threshold. Flagged by available screening inputs."
            )

        # Triage Rule 2: High uncertainty and missing critical inputs -> REQUEST ADDITIONAL TEST
        elif u > self.thresholds.max_acceptable_uncertainty and len(missing) > 0:
            action = ClinicalAction.ACQUIRE_MISSING_TEST
            explanation = (
                f"Deferred decision: epistemic uncertainty {u:.2f} is too high to rule out "
                f"pathology safely. Missing tests: {', '.join(missing)}. Recommend acquiring next modality."
            )

        # Triage Rule 3: High disbelief and acceptable uncertainty -> ROUTINE CARE
        elif d >= self.thresholds.routine_disbelief_threshold and u <= self.thresholds.max_acceptable_uncertainty:
            action = ClinicalAction.ROUTINE_CARE
            explanation = (
                f"Routine care: evidence of normality d={d:.2f} satisfies safety margin "
                f"with controlled uncertainty u={u:.2f}."
            )

        # Triage Rule 4: Borderline cases -> Conservative safety tilt
        else:
            if b > 0.15 or len(missing) > 0:
                action = ClinicalAction.ACQUIRE_MISSING_TEST
                explanation = (
                    f"Inconclusive: borderline belief {b:.2f} and uncertainty {u:.2f}. "
                    f"Recommend acquiring missing test {', '.join(missing) if missing else 'repeat scan'}."
                )
            else:
                action = ClinicalAction.ROUTINE_CARE
                explanation = f"Routine care with clinical follow-up: low residual suspicion."

        return ScreeningResult(
            action=action,
            fused_opinion=fused_opinion,
            explanation=explanation,
            missing_modalities=missing,
            raw_expert_opinions=raw_opinions
        )
