"""
Data classifier for categorizing reasoning styles and error types.

Uses GLM4.5 API for accurate classification with local caching and
fallback to heuristic rules when API is unavailable.

From experiment plan:
- Reasoning styles (for correct responses): ALGEBRAIC, DIRECT, BACKTRACKING
- Error types (for incorrect responses): ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.society.agent_registry import ReasoningStyle, ErrorType

logger = logging.getLogger(__name__)


# ============================================================
# Data classes
# ============================================================

@dataclass
class ReasoningStyleResult:
    """Result of reasoning style classification."""
    style: ReasoningStyle
    confidence: float
    raw_response: str = ""


@dataclass
class ErrorTypeResult:
    """Result of error type classification."""
    error_type: ErrorType
    confidence: float
    raw_response: str = ""


# ============================================================
# API configuration (from experiment plan)
# ============================================================

DEFAULT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_API_KEY = os.environ.get("GLM_API_KEY", "")
DEFAULT_API_MODEL = "glm-4-flash"

# Classification prompts (from experiment plan)
REASONING_STYLE_PROMPT = """Given a math problem and a correct solution, classify the reasoning style:

Problem: {question}
Solution: {response}
Correct Answer: {answer}

Classify into exactly one category:
- ALGEBRAIC: Uses symbolic manipulation, equations, variables (e.g., "let x =", solving systems)
- DIRECT: Direct step-by-step numerical computation without symbolic setup
- BACKTRACKING: Starts with an attempt, verifies it, then revises if needed

Respond with only the category name."""

ERROR_TYPE_PROMPT = """Given a math problem, an incorrect solution, and the correct answer,
classify the primary error type:

Problem: {question}
Incorrect Solution: {response}
Extracted Answer: {extracted_answer}
Correct Answer: {correct_answer}

Classify into exactly one category:
- ARITHMETIC: Correct reasoning approach but numerical calculation mistake
- LOGIC: Flawed reasoning chain, wrong formula, or logical fallacy
- HALLUCINATION: Fabricated numbers, wrong theorem, or unsupported claims
- VERIFICATION: Attempted self-check but failed to catch the error

Respond with only the category name."""


# ============================================================
# API call with retry
# ============================================================

def _call_api(
    prompt: str,
    api_url: str = DEFAULT_API_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_API_MODEL,
) -> Optional[str]:
    """Call GLM API for classification (OpenAI-compatible endpoint)."""
    if not api_key:
        logger.warning("GLM_API_KEY not set, falling back to heuristic classification")
        return None
    try:
        import requests

        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 64,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        return result
    except ImportError:
        logger.warning("requests package not installed, falling back to heuristic")
        return None
    except Exception as e:
        logger.warning(f"API call failed: {e}, falling back to heuristic")
        return None


# ============================================================
# Response parsers
# ============================================================

def _parse_style_response(response: str) -> Optional[ReasoningStyle]:
    """Parse API response into ReasoningStyle."""
    if not response:
        return None
    response_upper = response.upper().strip()
    for style in ReasoningStyle:
        if style.value.upper() in response_upper:
            return style
    return None


def _parse_error_response(response: str) -> Optional[ErrorType]:
    """Parse API response into ErrorType."""
    if not response:
        return None
    response_upper = response.upper().strip()
    for et in ErrorType:
        if et.value.upper() in response_upper:
            return et
    return None


# ============================================================
# Cache helpers
# ============================================================

def _compute_sample_hash(question: str, response: str) -> str:
    """Compute a hash for caching."""
    import hashlib
    content = f"{question}||{response}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _load_cache(cache_path: Path) -> Optional[dict]:
    """Load cached classification result."""
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_cache(cache_path: Path, data: dict) -> None:
    """Save classification result to cache (atomic write)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    tmp_path.rename(cache_path)


# ============================================================
# Public classification functions
# ============================================================

def classify_reasoning_style(
    response: str,
    question: str,
    correct_answer: str = "",
    use_api: bool = True,
    cache_dir: str = "cache/society/classifications",
) -> ReasoningStyleResult:
    """
    Classify the reasoning style of a correct response.

    Uses GLM4.5 API with local caching, falls back to heuristic.
    """
    sample_hash = _compute_sample_hash(question, response)
    cache_path = Path(cache_dir) / f"style_{sample_hash}.json"

    # Check cache
    cached = _load_cache(cache_path)
    if cached:
        return ReasoningStyleResult(
            style=ReasoningStyle(cached["style"]),
            confidence=cached.get("confidence", 0.9),
            raw_response=cached.get("raw_response", ""),
        )

    # API classification
    if use_api:
        prompt = REASONING_STYLE_PROMPT.format(
            question=question,
            response=response[:2000],
            answer=correct_answer,
        )
        api_response = _call_api(prompt)
        style = _parse_style_response(api_response)

        if style is not None:
            _save_cache(cache_path, {
                "style": style.value,
                "confidence": 0.9,
                "raw_response": api_response or "",
            })
            return ReasoningStyleResult(
                style=style, confidence=0.9, raw_response=api_response or ""
            )

    # Heuristic fallback
    style = _heuristic_style_classify(response)
    return ReasoningStyleResult(style=style, confidence=0.5, raw_response="(heuristic)")


def classify_error_type(
    response: str,
    question: str,
    extracted_answer: str = "",
    correct_answer: str = "",
    use_api: bool = True,
    cache_dir: str = "cache/society/classifications",
) -> ErrorTypeResult:
    """
    Classify the error type of an incorrect response.

    Uses GLM4.5 API with local caching, falls back to heuristic.
    """
    sample_hash = _compute_sample_hash(question, response)
    cache_path = Path(cache_dir) / f"error_{sample_hash}.json"

    # Check cache
    cached = _load_cache(cache_path)
    if cached:
        return ErrorTypeResult(
            error_type=ErrorType(cached["error_type"]),
            confidence=cached.get("confidence", 0.9),
            raw_response=cached.get("raw_response", ""),
        )

    # API classification
    if use_api:
        prompt = ERROR_TYPE_PROMPT.format(
            question=question,
            response=response[:2000],
            extracted_answer=extracted_answer,
            correct_answer=correct_answer,
        )
        api_response = _call_api(prompt)
        error_type = _parse_error_response(api_response)

        if error_type is not None:
            _save_cache(cache_path, {
                "error_type": error_type.value,
                "confidence": 0.9,
                "raw_response": api_response or "",
            })
            return ErrorTypeResult(
                error_type=error_type, confidence=0.9, raw_response=api_response or ""
            )

    # Heuristic fallback
    error_type = _heuristic_error_classify(response)
    return ErrorTypeResult(error_type=error_type, confidence=0.5, raw_response="(heuristic)")


# ============================================================
# Heuristic fallbacks
# ============================================================

def _heuristic_style_classify(response: str) -> ReasoningStyle:
    """Heuristic reasoning style classification."""
    lower = response.lower()
    algebraic_kw = ["let x", "let y", "equation", "variable", "substitut",
                     "system of equations", "algebraic", "symbolic"]
    backtracking_kw = ["verify", "check", "incorrect", "wrong", "let me try",
                        "actually", "instead", "revise", "retry"]

    a_score = sum(1 for kw in algebraic_kw if kw in lower)
    b_score = sum(1 for kw in backtracking_kw if kw in lower)

    if a_score > b_score and a_score > 0:
        return ReasoningStyle.ALGEBRAIC
    elif b_score > 0:
        return ReasoningStyle.BACKTRACKING
    return ReasoningStyle.DIRECT


def _heuristic_error_classify(response: str) -> ErrorType:
    """Heuristic error type classification."""
    lower = response.lower()
    if any(kw in lower for kw in ["fabricat", "invent", "wrong theorem", "unsupported"]):
        return ErrorType.HALLUCINATION
    if any(kw in lower for kw in ["logical fallacy", "non sequitur", "wrong approach"]):
        return ErrorType.LOGIC
    if any(kw in lower for kw in ["check", "verify", "self-check", "should have"]):
        return ErrorType.VERIFICATION
    return ErrorType.ARITHMETIC


# ============================================================
# Convenience class
# ============================================================

class DataClassifier:
    """Convenience wrapper for classification functions."""

    def classify_reasoning_style(
        self, response: str, question: str, correct_answer: str = "",
        use_api: bool = True, cache_dir: str = "cache/society/classifications",
    ) -> ReasoningStyleResult:
        return classify_reasoning_style(
            response, question, correct_answer, use_api, cache_dir
        )

    def classify_error_type(
        self, response: str, question: str, extracted_answer: str = "",
        correct_answer: str = "", use_api: bool = True,
        cache_dir: str = "cache/society/classifications",
    ) -> ErrorTypeResult:
        return classify_error_type(
            response, question, extracted_answer, correct_answer, use_api, cache_dir
        )
