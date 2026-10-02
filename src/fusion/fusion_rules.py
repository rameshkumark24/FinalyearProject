"""
SFR Framework — Fusion Rules Benchmark
======================================
Implements comparative fusion methods to benchmark against Subjective Logic OR:
1. Subjective Logic OR (Proposed)
2. Max Rule: max(p1, p2, p3)
3. Noisy-OR Rule: 1 - prod(1 - pi)
4. Dempster-Shafer Combination: Orthogonal sum with conflict normalization
5. Mean / Average Rule: mean(p_available)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from .subjective_logic import Opinion, fuse_opinions_or


class FusionBenchmark:
    """
    Comparison suite for multi-expert prenatal screening without paired data.
    """

    @staticmethod
    def max_rule(probabilities: List[Optional[float]]) -> float:
        """Baseline 1: Maximum predicted risk among available modalities."""
        valid_probs = [p for p in probabilities if p is not None and not np.isnan(p)]
        if not valid_probs:
            return 0.0
        return float(np.max(valid_probs))

    @staticmethod
    def noisy_or(probabilities: List[Optional[float]]) -> float:
        """Baseline 2: Probabilistic Noisy-OR: 1 - prod(1 - pi)."""
        valid_probs = [p for p in probabilities if p is not None and not np.isnan(p)]
        if not valid_probs:
            return 0.0
        prod = 1.0
        for p in valid_probs:
            prod *= (1.0 - max(0.0, min(1.0, p)))
        return float(1.0 - prod)

    @staticmethod
    def mean_rule(probabilities: List[Optional[float]]) -> float:
        """Baseline 3: Average predicted risk among available modalities."""
        valid_probs = [p for p in probabilities if p is not None and not np.isnan(p)]
        if not valid_probs:
            return 0.0
        return float(np.mean(valid_probs))

    @staticmethod
    def dempster_shafer(opinions: List[Optional[Opinion]]) -> Tuple[float, float, float]:
        """
        Baseline 4: Standard Dempster-Shafer rule of combination.
        Demonstrates the classic conflict problem (Zadeh's paradox in medicine):
        When structure expert flags CHD (high belief) but rhythm expert sees normal sinus rhythm
        (high disbelief of arrhythmia), DS normalizes away the conflict and suppresses the alarm!
        """
        valid_ops = [op for op in opinions if op is not None]
        if not valid_ops:
            return 0.0, 0.0, 1.0

        # Start with first opinion: m({A}), m({not A}), m(Theta)
        m_A = valid_ops[0].belief
        m_notA = valid_ops[0].disbelief
        m_theta = valid_ops[0].uncertainty

        for op in valid_ops[1:]:
            m2_A = op.belief
            m2_notA = op.disbelief
            m2_theta = op.uncertainty

            # Conflict: A & not A
            k = m_A * m2_notA + m_notA * m2_A
            if k >= 0.9999:  # Total conflict
                k = 0.9999

            norm = 1.0 - k

            # Combined masses
            new_m_A = (m_A * m2_A + m_A * m2_theta + m_theta * m2_A) / norm
            new_m_notA = (m_notA * m2_notA + m_notA * m2_theta + m_theta * m2_notA) / norm
            new_m_theta = (m_theta * m2_theta) / norm

            m_A, m_notA, m_theta = new_m_A, new_m_notA, new_m_theta

        return float(m_A), float(m_notA), float(m_theta)

    @staticmethod
    def subjective_logic(opinions: List[Optional[Opinion]]) -> Opinion:
        """Proposed: Subjective Logic Disjunctive (OR) Combination."""
        clean_ops = []
        for op in opinions:
            if op is None:
                clean_ops.append(Opinion.vacuous())
            else:
                clean_ops.append(op)
        return fuse_opinions_or(clean_ops)
