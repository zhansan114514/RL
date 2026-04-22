"""
MoE-style Critic Router for selecting and weighting Critic feedback.

No trainable parameters — routes based on Critic confidence scores
using softmax weighting and Top-K selection.

From experiment plan:
  Input:  All Critic (confidence, feedback_text)
  Process: softmax(confidence, temperature=1.0) → Top-K (default K=2) → weighted concat
  Output: Weighted feedback text + routing weights
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.society.agent_registry import AgentConfig, ErrorType

logger = logging.getLogger(__name__)


# ============================================================
# Data classes for routing
# ============================================================

@dataclass
class CriticFeedback:
    """Feedback from a single Critic."""
    critic_name: str
    error_specialty: ErrorType
    feedback_text: str
    confidence: float  # 0.0 to 1.0
    raw_response: str = ""


@dataclass
class RoutedFeedback:
    """Aggregated feedback after routing."""
    feedback_text: str
    selected_critics: list[str]
    weights: list[float]
    raw_feedbacks: list[CriticFeedback]


# ============================================================
# Confidence parser
# ============================================================

CONFIDENCE_PATTERN = re.compile(
    r'\[Confidence:\s*([0-9]*\.?[0-9]+)\]',
    re.IGNORECASE,
)


def parse_confidence(response: str) -> Optional[float]:
    """Parse [Confidence: 0.X] from Critic response."""
    match = CONFIDENCE_PATTERN.search(response)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return None
    return None


# ============================================================
# Critic Router (MoE-style, no trainable params)
# ============================================================

class CriticRouter:
    """
    Routes Critic feedback using confidence-based softmax weighting.

    Algorithm (from experiment plan):
    1. Each Critic outputs feedback + [Confidence: X]
    2. Parse confidence scores
    3. softmax(temperature) → routing weights
    4. Select Top-K Critics
    5. Weighted concatenation of feedback
    """

    def __init__(
        self,
        top_k: int = 2,
        temperature: float = 1.0,
        min_confidence: float = 0.1,
        fallback_to_uniform: bool = True,
        uniform_weights: bool = False,
    ):
        self.top_k = top_k
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.fallback_to_uniform = fallback_to_uniform
        self.uniform_weights = uniform_weights

    def route(self, feedbacks: list[CriticFeedback]) -> RoutedFeedback:
        """Route feedbacks to produce aggregated weighted feedback."""
        if not feedbacks:
            return RoutedFeedback("", [], [], [])

        # Filter by minimum confidence
        valid = [f for f in feedbacks if f.confidence >= self.min_confidence]
        if not valid:
            if self.fallback_to_uniform:
                valid = feedbacks
            else:
                return RoutedFeedback("", [], [], feedbacks)

        # Softmax on confidence (or uniform if uniform_weights=True)
        if self.uniform_weights:
            weights = np.ones(len(valid)) / len(valid)
        else:
            confidences = np.array([f.confidence for f in valid])
            weights = self._softmax(confidences, temperature=self.temperature)

        # Top-K selection
        k = min(self.top_k, len(valid))
        top_indices = np.argsort(weights)[-k:]
        selected = [valid[i] for i in top_indices]
        sel_weights = weights[top_indices]
        sel_weights = sel_weights / sel_weights.sum()

        # Weighted concatenation
        parts = []
        for fb, w in zip(selected, sel_weights):
            header = (f"[{fb.critic_name} | weight={w:.2f} | "
                      f"specialty={fb.error_specialty.value}]")
            parts.append(f"{header}\n{fb.feedback_text}")

        return RoutedFeedback(
            feedback_text="\n\n---\n\n".join(parts),
            selected_critics=[fb.critic_name for fb in selected],
            weights=sel_weights.tolist(),
            raw_feedbacks=feedbacks,
        )

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float = 0) -> np.ndarray:
        """Softmax with temperature scaling."""
        t = temperature if temperature > 0 else 1.0
        scaled = scores / t
        exp_s = np.exp(scaled - np.max(scaled))
        return exp_s / exp_s.sum()


def build_critic_feedback(
    critic_config: AgentConfig,
    response: str,
) -> CriticFeedback:
    """Build CriticFeedback from raw response, parsing confidence."""
    confidence = parse_confidence(response)
    if confidence is None:
        confidence = 0.5
        logger.debug(f"No confidence parsed for {critic_config.name}, default=0.5")

    clean_response = CONFIDENCE_PATTERN.sub("", response).strip()

    return CriticFeedback(
        critic_name=critic_config.name,
        error_specialty=critic_config.error_specialty,
        feedback_text=clean_response,
        confidence=confidence,
        raw_response=response,
    )
