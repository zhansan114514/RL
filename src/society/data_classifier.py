"""
Data classifier for categorizing reasoning styles and error types.

Uses GLM-4-flash API for classification with local caching.
Raises ClassificationError when API is required but unavailable,
instead of silently falling back to unreliable heuristics.

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
# Exceptions
# ============================================================

class ClassificationError(Exception):
    """Raised when classification fails (API unavailable, parse error, etc.)."""
    pass


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
# API availability check
# ============================================================

def check_api_available(
    api_url: str = DEFAULT_API_URL,
    api_key: str = DEFAULT_API_KEY,
) -> tuple[bool, str]:
    """Check if the GLM API is reachable and the key is configured.

    Returns:
        (available, reason) tuple.  `available` is True when the API
        can be used for classification.  `reason` describes the problem
        when unavailable.
    """
    if not api_key:
        return False, "GLM_API_KEY environment variable is not set"
    try:
        import requests
    except ImportError:
        return False, "requests package is not installed"

    try:
        resp = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_API_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        # Accept both 200 (success) and 4xx (auth/rate-limit but server is reachable)
        # A connection error would have thrown before this point.
        if resp.status_code == 401:
            return False, f"API key is invalid (HTTP 401)"
        return True, ""
    except requests.exceptions.ConnectionError as e:
        return False, f"Cannot connect to API: {e}"
    except requests.exceptions.Timeout:
        return False, "API connection timed out"
    except Exception as e:
        return False, f"API check failed: {e}"


# ============================================================
# API call with retry
# ============================================================

def _call_api(
    prompt: str,
    api_url: str = DEFAULT_API_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_API_MODEL,
) -> str:
    """Call GLM API for classification (OpenAI-compatible endpoint).

    Raises ClassificationError on any failure instead of silently
    returning None.
    """
    if not api_key:
        raise ClassificationError(
            "GLM_API_KEY not set. Set it via: export GLM_API_KEY=your_key"
        )
    try:
        import requests
    except ImportError:
        raise ClassificationError(
            "requests package not installed. Install via: pip install requests"
        )

    try:
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
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            raise ClassificationError("API returned null content (possible content filter)")
        result = content.strip()
        return result
    except requests.exceptions.ConnectionError as e:
        raise ClassificationError(f"Cannot connect to API: {e}") from e
    except requests.exceptions.Timeout:
        raise ClassificationError("API request timed out")
    except requests.exceptions.HTTPError as e:
        raise ClassificationError(f"API HTTP error: {e}") from e
    except (KeyError, IndexError, AttributeError) as e:
        raise ClassificationError(f"Unexpected API response format: {e}") from e
    except Exception as e:
        raise ClassificationError(f"API call failed: {e}") from e


# ============================================================
# Response parsers
# ============================================================

def _parse_style_response(response: str) -> Optional[ReasoningStyle]:
    """Parse API response into ReasoningStyle.

    Handles markdown bold, word variants, and verbose responses.
    Uses word-boundary matching to avoid false positives like
    'NOT ALGEBRAIC' matching ALGEBRAIC.
    """
    if not response:
        return None
    import re
    # Strip markdown bold/italic, quotes, brackets
    clean = re.sub(r'[*_`"\'\(\)\[\]{}]', ' ', response.upper().strip())
    for style in ReasoningStyle:
        pattern = r'\b' + re.escape(style.value.upper()) + r'\b'
        if re.search(pattern, clean):
            return style
    # Variant matching (e.g., ALGEBRAICALLY -> ALGEBRAIC)
    variants = {
        "ALGEBRAICALLY": ReasoningStyle.ALGEBRAIC,
        "ALGEBRA": ReasoningStyle.ALGEBRAIC,
        "DIRECTLY": ReasoningStyle.DIRECT,
        "BACKTRACK": ReasoningStyle.BACKTRACKING,
    }
    words = re.findall(r'\b[A-Z]+\b', clean)
    for word in words:
        if word in variants:
            return variants[word]
    return None


def _parse_error_response(response: str) -> Optional[ErrorType]:
    """Parse API response into ErrorType.

    Handles markdown bold, word variants ("LOGICAL" -> LOGIC),
    and verbose responses.
    """
    if not response:
        return None
    import re
    # Strip markdown bold/italic, quotes, brackets
    clean = re.sub(r'[*_`"\'\(\)\[\]{}]', ' ', response.upper().strip())
    for et in ErrorType:
        pattern = r'\b' + re.escape(et.value.upper()) + r'\b'
        if re.search(pattern, clean):
            return et
    # Variant matching (e.g., LOGICAL -> LOGIC)
    variants = {
        "LOGICAL": ErrorType.LOGIC,
        "LOGICALLY": ErrorType.LOGIC,
        "ARITHMETICAL": ErrorType.ARITHMETIC,
        "HALLUCINATE": ErrorType.HALLUCINATION,
        "HALLUCINATING": ErrorType.HALLUCINATION,
        "VERIFY": ErrorType.VERIFICATION,
        "VERIFYING": ErrorType.VERIFICATION,
    }
    words = re.findall(r'\b[A-Z]+\b', clean)
    for word in words:
        if word in variants:
            return variants[word]
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
    cache_dir: str = "output/society/classified",
) -> ReasoningStyleResult:
    """
    Classify the reasoning style of a correct response.

    Uses GLM API with local caching.
    Raises ClassificationError if use_api=True but API is unavailable
    and no cached result exists.
    """
    sample_hash = _compute_sample_hash(question, response)
    cache_path = Path(cache_dir) / f"style_{sample_hash}.json"

    # Check cache first (always succeeds regardless of API)
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

        # API responded but we couldn't parse it
        raise ClassificationError(
            f"Could not parse reasoning style from API response: {api_response!r}"
        )

    # use_api=False: fall back to deterministic assignment
    import hashlib
    styles = list(ReasoningStyle)
    idx = int(hashlib.md5((question + response).encode()).hexdigest(), 16) % len(styles)
    fallback = styles[idx]
    return ReasoningStyleResult(style=fallback, confidence=0.5, raw_response="fallback")


def classify_error_type(
    response: str,
    question: str,
    extracted_answer: str = "",
    correct_answer: str = "",
    use_api: bool = True,
    cache_dir: str = "output/society/classified",
) -> ErrorTypeResult:
    """
    Classify the error type of an incorrect response.

    Uses GLM API with local caching.
    Raises ClassificationError if use_api=True but API is unavailable
    and no cached result exists.
    """
    sample_hash = _compute_sample_hash(question, response)
    cache_path = Path(cache_dir) / f"error_{sample_hash}.json"

    # Check cache first
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

        raise ClassificationError(
            f"Could not parse error type from API response: {api_response!r}"
        )

    # use_api=False: fall back to deterministic assignment
    import hashlib
    errors = list(ErrorType)
    idx = int(hashlib.md5((question + response).encode()).hexdigest(), 16) % len(errors)
    fallback = errors[idx]
    return ErrorTypeResult(error_type=fallback, confidence=0.5, raw_response="fallback")


# ============================================================
# Convenience class
# ============================================================

class DataClassifier:
    """Convenience wrapper for classification functions."""

    def classify_reasoning_style(
        self, response: str, question: str, correct_answer: str = "",
        use_api: bool = True, cache_dir: str = "output/society/classified",
    ) -> ReasoningStyleResult:
        return classify_reasoning_style(
            response, question, correct_answer, use_api, cache_dir
        )

    def classify_error_type(
        self, response: str, question: str, extracted_answer: str = "",
        correct_answer: str = "", use_api: bool = True,
        cache_dir: str = "output/society/classified",
    ) -> ErrorTypeResult:
        return classify_error_type(
            response, question, extracted_answer, correct_answer, use_api, cache_dir
        )
