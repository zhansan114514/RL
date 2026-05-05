"""
MoE-style Critic Router for selecting and weighting Critic feedback.

No trainable parameters - routes based on validated Critic confidence scores
using softmax weighting and Top-K selection.

Only schema-valid Critic verdicts are eligible for routing.  Missing or
contradictory structured fields receive zero effective confidence so malformed
feedback cannot silently become a useful 0.5-weight signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.society.agent_registry import AgentConfig, CriticSkill
from src.society.critic_schema import (
    parse_answer_correct,
    parse_confidence,
    parse_critic_judgement,
    parse_error_type,
    parse_suggested_answer,
    strip_critic_judgement_fields,
)

logger = logging.getLogger(__name__)


# ============================================================
# Data classes for routing
# ============================================================

@dataclass
class CriticFeedback:
    """Feedback from a single Critic."""
    critic_name: str
    error_specialty: CriticSkill
    feedback_text: str
    confidence: float  # 0.0 to 1.0
    answer_correct: Optional[bool] = None  # Critic's judgment on Actor answer correctness
    suggested_answer: Optional[str] = None  # Critic's suggested final answer
    error_type: Optional[str] = None  # Critic's identified error type
    raw_response: str = ""
    schema_valid: bool = True
    schema_errors: list[str] = None

    def __post_init__(self) -> None:
        if self.schema_errors is None:
            self.schema_errors = []


@dataclass
class RoutedFeedback:
    """Aggregated feedback after routing."""
    feedback_text: str
    selected_critics: list[str]
    weights: list[float]
    raw_feedbacks: list[CriticFeedback]
    used_uniform_fallback: bool = False


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
        fallback_to_uniform: bool = False,
        uniform_weights: bool = False,
    ):
        self.top_k = top_k
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.fallback_to_uniform = fallback_to_uniform
        self.uniform_weights = uniform_weights

    def route(self, feedbacks: list[CriticFeedback]) -> RoutedFeedback:
        """Route feedbacks to produce aggregated weighted feedback.

        Prioritises Critics that detected an error (Answer_Correct: no)
        so the Actor receives corrective feedback, not mere confirmations.
        """
        if not feedbacks:
            return RoutedFeedback("", [], [], [])

        # Only schema-valid Critics are eligible for routing.  Invalid verdicts
        # remain visible in raw_feedbacks for diagnostics, but never affect the
        # Actor prompt or consensus weighting.
        schema_valid = [f for f in feedbacks if f.schema_valid]
        if not schema_valid:
            return RoutedFeedback("", [], [], feedbacks)

        # Filter by minimum confidence
        valid = [f for f in schema_valid if f.confidence >= self.min_confidence]
        used_uniform_fallback = False
        if not valid:
            if self.fallback_to_uniform:
                valid = schema_valid
                used_uniform_fallback = True
            else:
                return RoutedFeedback("", [], [], feedbacks)

        # Prefer Critics that found errors — they are more actionable
        negative = [f for f in valid if f.answer_correct is False]
        if negative:
            # Only promote negative critics; still keep all valid for weighting
            pass  # selection below handles priority

        # Softmax on confidence (or uniform if uniform_weights=True)
        if self.uniform_weights:
            weights = np.ones(len(valid)) / len(valid)
        else:
            confidences = np.array([f.confidence for f in valid])
            weights = self._softmax(confidences, temperature=self.temperature)

        # Boost weights for error-detecting Critics so they are
        # preferentially selected by Top-K.
        if negative:
            for i, fb in enumerate(valid):
                if fb.answer_correct is False:
                    weights[i] *= 2.0
            # Renormalise
            weights = weights / weights.sum()

        # Top-K selection (descending order: highest weight first)
        k = min(self.top_k, len(valid))
        top_indices = np.argsort(weights)[-k:][::-1]
        selected = [valid[i] for i in top_indices]
        sel_weights = weights[top_indices]
        # Renormalize after Top-K so weights sum to 1.0
        # Note: these are relative weights within the selected subset,
        # not the original softmax probabilities.
        weight_sum = sel_weights.sum()
        sel_weights = sel_weights / weight_sum if weight_sum > 0 else np.ones_like(sel_weights) / len(sel_weights)

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
            used_uniform_fallback=used_uniform_fallback,
        )

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature scaling.  temperature must be > 0."""
        if temperature <= 0:
            logger.warning(f"Router temperature={temperature} <= 0, clamping to 1.0")
            temperature = 1.0
        scaled = scores / temperature
        exp_s = np.exp(scaled - np.max(scaled))
        return exp_s / exp_s.sum()


def build_critic_feedback(
    critic_config: AgentConfig,
    response: str,
    actor_answer: Optional[str] = None,
) -> CriticFeedback:
    """Build CriticFeedback from raw response, parsing confidence and answer correctness."""
    judgement = parse_critic_judgement(response, actor_answer=actor_answer)
    parsed_confidence = judgement.confidence
    answer_correct = judgement.answer_correct
    suggested_answer = parse_suggested_answer(response)
    error_type = judgement.error_type
    schema_valid = judgement.schema_valid
    schema_errors = judgement.schema_errors

    if parsed_confidence is None:
        logger.debug(f"No confidence parsed for {critic_config.name}; routing confidence=0.0")
    if not schema_valid:
        logger.debug(
            "Invalid critic verdict for %s; routing confidence=0.0; errors=%s",
            critic_config.name,
            schema_errors,
    )
    confidence = parsed_confidence if schema_valid and parsed_confidence is not None else 0.0

    # Strip structured verdict tags from displayed feedback text.
    clean_response = strip_critic_judgement_fields(response)

    error_specialty = critic_config.error_specialty
    if error_specialty is None:
        raise ValueError(f"Critic '{critic_config.name}' has no error_specialty set")

    return CriticFeedback(
        critic_name=critic_config.name,
        error_specialty=error_specialty,
        feedback_text=clean_response,
        confidence=confidence,
        answer_correct=answer_correct,
        suggested_answer=suggested_answer,
        error_type=error_type,
        raw_response=response,
        schema_valid=schema_valid,
        schema_errors=schema_errors,
    )
