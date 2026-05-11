"""Field-level Critic router for natural deliberation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.parsing.critic_parser import ParsedCritic, parse_critic_response
from src.society.agent_registry import AgentConfig

logger = logging.getLogger(__name__)


DEFAULT_FEEDBACK_SCORE = 0.35


@dataclass(frozen=True)
class CriticFeedback:
    """Parsed feedback from one Critic."""

    critic_name: str
    skill: str
    critique: str
    confidence: Optional[float]
    answer_correct: str
    suggested_answer: Optional[str]
    usable_for_feedback: bool
    usable_for_routing: bool
    usable_for_consensus: bool
    raw_response: str = ""
    parse_errors: list[str] = field(default_factory=list)

    @property
    def route_score(self) -> float:
        """Score used for feedback selection."""
        if self.confidence is not None:
            return self.confidence
        if self.usable_for_feedback:
            return DEFAULT_FEEDBACK_SCORE
        return 0.0


@dataclass(frozen=True)
class RoutingDecision:
    """Aggregated natural feedback after routing."""

    feedback_text: str
    selected_feedbacks: list[CriticFeedback]
    weights: list[float]


class CriticRouter:
    """Select Top-K natural Critic feedback using field-level parse signals."""

    def __init__(
        self,
        top_k: int = 2,
        temperature: float = 1.0,
        min_confidence: float = 0.0,
        fallback_to_uniform: bool = False,
        uniform_weights: bool = False,
        default_score: float = DEFAULT_FEEDBACK_SCORE,
    ):
        self.top_k = top_k
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.fallback_to_uniform = fallback_to_uniform
        self.uniform_weights = uniform_weights
        self.default_score = default_score

    def route(self, feedbacks: list[CriticFeedback]) -> RoutingDecision:
        """Route usable natural critiques without exposing weights to Actors."""
        if not feedbacks:
            return RoutingDecision("", [], [])

        valid = [fb for fb in feedbacks if fb.usable_for_feedback]
        if not valid:
            return RoutingDecision("", [], [])

        scored = [
            fb for fb in valid
            if (fb.confidence is None and self._route_score(fb) > 0)
            or (fb.confidence is not None and fb.confidence >= self.min_confidence)
        ]
        if not scored:
            if self.fallback_to_uniform:
                scored = valid
            else:
                return RoutingDecision("", [], [])

        # Corrective feedback is usually more actionable than confirmation.
        corrective = [fb for fb in scored if fb.answer_correct == "no"]
        selection_pool = corrective or scored

        weights = self._weights(selection_pool)
        k = min(self.top_k, len(selection_pool))
        top_indices = np.argsort(weights)[-k:][::-1]
        selected = [selection_pool[i] for i in top_indices]
        selected_weights = weights[top_indices]
        total = float(selected_weights.sum())
        if total > 0:
            selected_weights = selected_weights / total
        else:
            selected_weights = np.ones(len(selected)) / max(len(selected), 1)

        parts = [
            f"{idx}. {fb.critique.strip()}"
            for idx, fb in enumerate(selected, start=1)
            if fb.critique.strip()
        ]
        return RoutingDecision(
            feedback_text="\n\n".join(parts),
            selected_feedbacks=selected,
            weights=selected_weights.tolist(),
        )

    def _weights(self, feedbacks: list[CriticFeedback]) -> np.ndarray:
        if not feedbacks:
            return np.array([])
        if self.uniform_weights:
            return np.ones(len(feedbacks)) / len(feedbacks)
        scores = np.array([self._route_score(fb) for fb in feedbacks], dtype=float)
        return self._softmax(scores, self.temperature)

    def _route_score(self, feedback: CriticFeedback) -> float:
        if feedback.confidence is not None:
            return feedback.confidence
        if feedback.usable_for_feedback:
            return self.default_score
        return 0.0

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if len(scores) == 0:
            return scores
        if temperature <= 0:
            logger.warning("Router temperature=%s <= 0, clamping to 1.0", temperature)
            temperature = 1.0
        scaled = scores / temperature
        exp_s = np.exp(scaled - np.max(scaled))
        denom = exp_s.sum()
        return exp_s / denom if denom > 0 else np.ones(len(scores)) / len(scores)


def build_critic_feedback(
    critic_config: AgentConfig,
    response: str,
    task_type: str = "multiple_choice",
) -> CriticFeedback:
    """Build field-level CriticFeedback from a raw Critic response."""
    parsed: ParsedCritic = parse_critic_response(response, task_type=task_type)
    skill = critic_config.error_specialty.value if critic_config.error_specialty else critic_config.name
    return CriticFeedback(
        critic_name=critic_config.name,
        skill=skill,
        critique=parsed.critique,
        confidence=parsed.confidence,
        answer_correct=parsed.answer_correct,
        suggested_answer=parsed.suggested_answer,
        usable_for_feedback=parsed.usable_for_feedback,
        usable_for_routing=parsed.usable_for_routing,
        usable_for_consensus=parsed.usable_for_consensus,
        raw_response=response,
        parse_errors=parsed.parse_errors,
    )
