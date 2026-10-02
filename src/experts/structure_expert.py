"""
SFR Framework — Structure Expert (Fetal Echocardiography)
=========================================================
Team Member: Rameshkumar
Novelty:
1. First evaluation of FetalCLIP (Maani 2026) on public CHD data across trimesters (2T vs 3T).
2. Parameter-efficient LoRA adapter on FetalCLIP image encoder.
3. Multi-view aggregation (4CH, 3VT, LVOT, RVOT) with view-dropout robustness.
4. Comparative controls: Heart-ViT (reproduced benchmark), DINOv2, BiomedCLIP, ResNet-50.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from ..fusion.subjective_logic import Opinion


STANDARD_ECHO_VIEWS = ["4CH", "3VT", "LVOT", "RVOT"]


class ViewAggregator:
    """
    Combines feature representations or prediction probabilities across multiple standard echo views.
    Handles missing views gracefully.
    """

    @staticmethod
    def aggregate_probabilities(
        view_probs: Dict[str, Optional[float]],
        method: str = "mean"
    ) -> Tuple[float, float]:
        """
        Aggregate predicted risk across available echocardiography views.
        Returns: (aggregated_probability, view_coverage_ratio)
        """
        valid_probs = [p for v, p in view_probs.items() if p is not None and not np.isnan(p)]
        coverage = len(valid_probs) / float(len(STANDARD_ECHO_VIEWS))

        if not valid_probs:
            return 0.009, 0.0  # Population prior, zero coverage

        if method == "max":
            # If any anatomical plane reveals a structural defect, preserve alarm
            agg_p = float(np.max(valid_probs))
        elif method == "weighted":
            # 4CH and 3VT typically carry highest diagnostic sensitivity in prenatal screening
            weights = {"4CH": 0.35, "3VT": 0.35, "LVOT": 0.15, "RVOT": 0.15}
            numerator = sum(view_probs[v] * weights.get(v, 0.25) for v in view_probs if view_probs[v] is not None)
            denom = sum(weights.get(v, 0.25) for v in view_probs if view_probs[v] is not None)
            agg_p = float(numerator / denom) if denom > 0 else float(np.mean(valid_probs))
        else:  # 'mean'
            agg_p = float(np.mean(valid_probs))

        return agg_p, coverage


class StructureExpert:
    """
    Expert evaluating fetal cardiac anatomical structures for Congenital Heart Defects (CHD).
    """

    def __init__(
        self,
        backbone_name: str = "FetalCLIP",  # 'FetalCLIP', 'Heart-ViT', 'DINOv2', 'BiomedCLIP', 'ResNet50'
        use_lora: bool = True,
        aggregation_method: str = "weighted",
        temperature: float = 1.0
    ):
        self.backbone_name = backbone_name
        self.use_lora = use_lora
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        self.aggregator = ViewAggregator()

    def evaluate(
        self,
        view_probabilities: Dict[str, Optional[float]],
        image_quality_scores: Optional[Dict[str, float]] = None
    ) -> Tuple[Opinion, Dict[str, Any]]:
        """
        Synthesize per-view predictions into a calibrated Subjective Logic Opinion (b, d, u).
        
        Args:
            view_probabilities: Dict mapping view name ('4CH', '3VT', 'LVOT', 'RVOT') to CHD probability.
            image_quality_scores: Optional acoustic shadow / quality score per view in [0, 1].
        """
        agg_prob, coverage = self.aggregator.aggregate_probabilities(
            view_probabilities, method=self.aggregation_method
        )

        # Temperature scaling
        if self.temperature != 1.0 and 0.0 < agg_prob < 1.0:
            logit = np.log(agg_prob / (1.0 - agg_prob)) / self.temperature
            calibrated_prob = float(1.0 / (1.0 + np.exp(-logit)))
        else:
            calibrated_prob = agg_prob

        # Quality weighting across available views
        if image_quality_scores:
            valid_qualities = [image_quality_scores[v] for v in view_probabilities if view_probabilities[v] is not None and v in image_quality_scores]
            mean_quality = float(np.mean(valid_qualities)) if valid_qualities else 0.85
        else:
            mean_quality = 0.85

        # Epistemic confidence depends on both image clarity AND anatomical coverage
        # (e.g. 1 view out of 4 gives lower confidence than all 4 standard planes)
        coverage_factor = 0.50 + 0.50 * coverage  # Range [0.5, 1.0] if at least 1 view present
        if coverage == 0.0:
            return Opinion.vacuous(base_rate=0.009), {"coverage": 0.0, "calibrated_prob": 0.009}

        confidence = float(np.clip(mean_quality * coverage_factor, 0.1, 0.95))

        opinion = Opinion.from_probability_and_confidence(
            prob=calibrated_prob,
            confidence=confidence,
            base_rate=0.009
        )

        metadata = {
            "backbone": self.backbone_name,
            "raw_view_probs": view_probabilities,
            "calibrated_prob": calibrated_prob,
            "view_coverage": coverage,
            "confidence": confidence
        }

        return opinion, metadata
