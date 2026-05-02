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
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.society.agent_registry import AgentConfig, CriticSkill

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


# ============================================================
# Confidence parser
# ============================================================

CONFIDENCE_PATTERN = re.compile(
    r'\[Confidence:\s*([0-9]*\.?[0-9]+)\]',
    re.IGNORECASE,
)

ANSWER_CORRECT_PATTERN = re.compile(
    r'\[Answer_Correct:\s*(yes|no)\]',
    re.IGNORECASE,
)

SUGGESTED_ANSWER_PATTERN = re.compile(
    r'\[Suggested_Final_Answer:\s*([A-Da-d]|Yes|No|unknown)\]',
    re.IGNORECASE,
)

ERROR_TYPE_PATTERN = re.compile(
    r'\[Error_Type:\s*(arithmetic|logic|hallucination|verification|none)\]',
    re.IGNORECASE,
)

CRITIC_VERDICT_PATTERNS = [
    ("answer_correct", ANSWER_CORRECT_PATTERN),
    ("suggested_answer", SUGGESTED_ANSWER_PATTERN),
    ("error_type", ERROR_TYPE_PATTERN),
    ("confidence", CONFIDENCE_PATTERN),
]


def parse_confidence(response: str) -> Optional[float]:
    """Parse [Confidence: 0.X] from Critic response."""
    match = CONFIDENCE_PATTERN.search(response)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return None
    return None


def parse_answer_correct(response: str) -> Optional[bool]:
    """Parse [Answer_Correct: yes/no] from Critic response."""
    match = ANSWER_CORRECT_PATTERN.search(response)
    if match:
        return match.group(1).lower() == "yes"
    return None


def parse_suggested_answer(response: str) -> Optional[str]:
    """Parse [Suggested_Final_Answer: X] from Critic response."""
    match = SUGGESTED_ANSWER_PATTERN.search(response)
    if not match:
        return None
    ans = match.group(1).strip()
    if ans.lower() == "unknown":
        return None
    if ans.upper() in {"A", "B", "C", "D"}:
        return ans.upper()
    if ans.lower() in {"yes", "no"}:
        return ans.upper()
    return ans


def parse_error_type(response: str) -> Optional[str]:
    """Parse [Error_Type: X] from Critic response."""
    match = ERROR_TYPE_PATTERN.search(response)
    if match:
        return match.group(1).lower()
    return None


def _normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    normalized = str(answer).strip()
    if not normalized:
        return None
    upper = normalized.upper()
    if upper in {"A", "B", "C", "D", "YES", "NO"}:
        return upper
    return normalized


def _validate_critic_schema(
    response: str,
    actor_answer: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Validate that the Critic response starts with a consistent verdict block."""
    errors: list[str] = []
    lines = [line.strip() for line in response.splitlines() if line.strip()]

    if len(lines) < len(CRITIC_VERDICT_PATTERNS):
        errors.append("missing_verdict_block")
    else:
        for idx, (field_name, pattern) in enumerate(CRITIC_VERDICT_PATTERNS):
            if not pattern.fullmatch(lines[idx]):
                errors.append(f"{field_name}_not_in_verdict_position")

    if parse_answer_correct(response) is None:
        errors.append("missing_answer_correct")
    if SUGGESTED_ANSWER_PATTERN.search(response) is None:
        errors.append("missing_suggested_answer")
    if parse_error_type(response) is None:
        errors.append("missing_error_type")
    if parse_confidence(response) is None:
        errors.append("missing_confidence")

    answer_correct = parse_answer_correct(response)
    suggested_answer = _normalize_answer(parse_suggested_answer(response))
    actor_answer = _normalize_answer(actor_answer)

    if actor_answer and suggested_answer:
        if answer_correct is True and suggested_answer != actor_answer:
            errors.append("contradiction_correct_with_different_suggestion")
        elif answer_correct is False and suggested_answer == actor_answer:
            errors.append("contradiction_incorrect_with_same_suggestion")

    # Deduplicate while preserving order for readable traces.
    unique_errors = list(dict.fromkeys(errors))
    return not unique_errors, unique_errors


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
        if not valid:
            if self.fallback_to_uniform:
                valid = schema_valid
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
    parsed_confidence = parse_confidence(response)
    answer_correct = parse_answer_correct(response)
    suggested_answer = parse_suggested_answer(response)
    error_type = parse_error_type(response)
    schema_valid, schema_errors = _validate_critic_schema(response, actor_answer=actor_answer)

    if parsed_confidence is None:
        logger.debug(f"No confidence parsed for {critic_config.name}; routing confidence=0.0")
    if not schema_valid:
        logger.debug(
            "Invalid critic verdict for %s; routing confidence=0.0; errors=%s",
            critic_config.name,
            schema_errors,
        )
    confidence = parsed_confidence if schema_valid and parsed_confidence is not None else 0.0

    # Strip all structured tags from the displayed feedback text
    clean_response = CONFIDENCE_PATTERN.sub("", response)
    clean_response = ANSWER_CORRECT_PATTERN.sub("", clean_response)
    clean_response = SUGGESTED_ANSWER_PATTERN.sub("", clean_response)
    clean_response = ERROR_TYPE_PATTERN.sub("", clean_response).strip()

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
