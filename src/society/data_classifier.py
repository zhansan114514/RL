"""
Data classifier for categorizing reasoning styles and critic error profiles.

Uses GLM-4-flash API for classification with local caching.
Raises ClassificationError when API is required but unavailable,
instead of silently falling back to unreliable heuristics.

From experiment plan:
- Reasoning styles (for correct responses): ALGEBRAIC, DIRECT, BACKTRACKING
- Error profiles (for incorrect responses): computation/reasoning/knowledge/grounding/verification
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.society.agent_registry import ReasoningStyle

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
class ErrorProfileResult:
    """Result of multi-dimensional error profile classification."""
    scores: dict[str, float]
    primary: str
    secondary: list[str]
    confidence: float
    evidence: str = ""
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

ERROR_PROFILE_PROMPT = """
You are classifying why a model response is wrong.

Dataset: {dataset_name}
Task type: {task_type}
Subject/domain: {subject}

Question:
{question}

Choices:
{choices}

Model response:
{response}

Extracted Answer: {extracted_answer}
Correct Answer: {correct_answer}

Score each error dimension from 0.0 to 1.0:

- computation: numerical calculation, algebra, symbolic manipulation, formula computation
- reasoning: flawed reasoning chain, invalid inference, wrong rule application
- knowledge: wrong factual/domain knowledge, concept confusion
- grounding: ignores or contradicts the question/options, invents unsupported assumptions
- verification: fails to check final answer, option-letter mismatch, self-check failure

Return JSON only:
{{
  "scores": {{
    "computation": float,
    "reasoning": float,
    "knowledge": float,
    "grounding": float,
    "verification": float
  }},
  "primary": "computation|reasoning|knowledge|grounding|verification|unknown",
  "secondary": ["computation|reasoning|knowledge|grounding|verification"],
  "confidence": float,
  "evidence": "short reason"
}}
"""

ERROR_PROFILE_DIMENSIONS = (
    "computation",
    "reasoning",
    "knowledge",
    "grounding",
    "verification",
)


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
            return False, "API key is invalid (HTTP 401)"
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
    max_tokens: int = 64,
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
                "max_tokens": max_tokens,
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


def _clamp01(value: object, default: float = 0.0) -> float:
    """Convert a value to a bounded 0..1 float."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _unknown_error_profile(raw_response: str = "", evidence: str = "") -> ErrorProfileResult:
    """Return an explicit unknown profile without assigning it to reasoning/logic."""
    return ErrorProfileResult(
        scores={dim: 0.0 for dim in ERROR_PROFILE_DIMENSIONS},
        primary="unknown",
        secondary=[],
        confidence=0.0,
        evidence=evidence,
        raw_response=raw_response,
    )


def _parse_error_profile_response(response: str) -> Optional[ErrorProfileResult]:
    """Parse a JSON error-profile API response."""
    if not response:
        return None

    import re

    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # GLM API returns LaTeX-style backslashes (e.g. \( \sqrt \))
        # which are invalid JSON escape sequences. Fix and retry.
        try:
            fixed = re.sub(
                r'\\([^"\\/bfnrtu])',
                lambda m: '\\\\' + m.group(1),
                text,
            )
            data = json.loads(fixed)
        except json.JSONDecodeError:
            # Last resort: strip evidence field entirely and retry
            try:
                stripped = re.sub(r'"evidence"\s*:\s*"[^"]*"', '"evidence": ""', text)
                stripped = re.sub(r'"evidence"\s*:\s*\{[^}]*\}', '"evidence": ""', stripped)
                data = json.loads(stripped)
            except json.JSONDecodeError:
                return None

    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {
        dim: _clamp01(raw_scores.get(dim), 0.0)
        for dim in ERROR_PROFILE_DIMENSIONS
    }

    primary = str(data.get("primary", "")).strip().lower()
    if primary not in ERROR_PROFILE_DIMENSIONS:
        primary = max(scores.items(), key=lambda kv: kv[1])[0] if any(scores.values()) else "unknown"

    secondary_raw = data.get("secondary", [])
    if isinstance(secondary_raw, str):
        secondary_raw = [secondary_raw]
    secondary = []
    if isinstance(secondary_raw, list):
        for label in secondary_raw:
            label = str(label).strip().lower()
            if label in ERROR_PROFILE_DIMENSIONS and label != primary and label not in secondary:
                secondary.append(label)

    confidence = _clamp01(data.get("confidence"), 0.5)
    evidence = str(data.get("evidence", "")).strip()

    return ErrorProfileResult(
        scores=scores,
        primary=primary,
        secondary=secondary,
        confidence=confidence,
        evidence=evidence,
        raw_response=response or "",
    )


# ============================================================
# Cache helpers
# ============================================================

def _compute_sample_hash(question: str, response: str) -> str:
    """Compute a hash for caching reasoning style (question + response only)."""
    import hashlib
    content = f"{question}||{response}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _compute_error_profile_hash(
    question: str,
    response: str,
    correct_answer: str = "",
    choices: str = "",
    dataset_name: str = "",
    task_type: str = "",
    subject: str = "",
) -> str:
    """Compute a hash for caching error profiles (includes full context).

    Unlike reasoning style, error profiles depend on the full problem context
    (choices, correct answer, dataset, task type, subject).  Using only
    question+response would cause cache pollution across datasets with
    different choices or task types.
    """
    import hashlib
    content = json.dumps({
        "dataset_name": dataset_name,
        "task_type": task_type,
        "subject": subject,
        "question": question,
        "choices": choices,
        "response": response,
        "correct_answer": correct_answer,
    }, sort_keys=True, ensure_ascii=False)
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
    import threading
    import uuid
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(
        f"{cache_path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, cache_path)


# ============================================================
# Public classification functions
# ============================================================

def classify_reasoning_style(
    response: str,
    question: str,
    correct_answer: str = "",
    use_api: bool = True,
    cache_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
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
        api_response = _call_api(
            prompt,
            api_url=api_base or DEFAULT_API_URL,
            api_key=api_key or DEFAULT_API_KEY,
            model=api_model or DEFAULT_API_MODEL,
        )
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


def classify_error_profile(
    response: str,
    question: str,
    extracted_answer: str = "",
    correct_answer: str = "",
    choices: str | list | dict = "",
    dataset_name: str = "",
    task_type: str = "",
    subject: str = "",
    use_api: bool = True,
    cache_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
) -> ErrorProfileResult:
    """
    Classify the multi-dimensional error profile of an incorrect response.

    Uses GLM API with local caching.  The cache key includes the full problem
    context (dataset_name, task_type, subject, choices, correct_answer) to
    avoid cache pollution across datasets with different contexts.

    Raises ClassificationError if use_api=True but API is unavailable
    and no cached result exists.
    """
    if isinstance(choices, (list, dict)):
        choices_text = json.dumps(choices, ensure_ascii=False)
    else:
        choices_text = str(choices or "")

    sample_hash = _compute_error_profile_hash(
        question=question,
        response=response,
        correct_answer=correct_answer,
        choices=choices_text,
        dataset_name=dataset_name,
        task_type=task_type,
        subject=subject,
    )
    cache_path = Path(cache_dir) / f"error_profile_{sample_hash}.json"

    # Check cache first
    cached = _load_cache(cache_path)
    if cached:
        return ErrorProfileResult(
            scores={dim: _clamp01(cached.get("scores", {}).get(dim), 0.0)
                    for dim in ERROR_PROFILE_DIMENSIONS},
            primary=cached.get("primary", "unknown"),
            secondary=cached.get("secondary", []),
            confidence=_clamp01(cached.get("confidence"), 0.0),
            evidence=cached.get("evidence", ""),
            raw_response=cached.get("raw_response", ""),
        )

    # API classification
    if use_api:
        prompt = ERROR_PROFILE_PROMPT.format(
            dataset_name=dataset_name or "unknown",
            task_type=task_type or "unknown",
            subject=subject or "unknown",
            question=question,
            choices=choices_text,
            response=response[:2000],
            extracted_answer=extracted_answer,
            correct_answer=correct_answer,
        )
        api_response = _call_api(
            prompt,
            api_url=api_base or DEFAULT_API_URL,
            api_key=api_key or DEFAULT_API_KEY,
            model=api_model or DEFAULT_API_MODEL,
            max_tokens=512,
        )
        profile = _parse_error_profile_response(api_response)

        if profile is not None:
            _save_cache(cache_path, {
                "scores": profile.scores,
                "primary": profile.primary,
                "secondary": profile.secondary,
                "confidence": profile.confidence,
                "evidence": profile.evidence,
                "raw_response": profile.raw_response,
            })
            return profile

        raise ClassificationError(
            f"Could not parse error profile JSON from API response: {api_response!r}"
        )

    return _unknown_error_profile(raw_response="fallback", evidence="use_api=False")


# ============================================================
# Convenience class
# ============================================================

class DataClassifier:
    """Convenience wrapper for classification functions."""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        api_model: str = "",
    ):
        self._api_key = api_key
        self._api_base = api_base
        self._api_model = api_model

    def classify_reasoning_style(
        self, response: str, question: str, correct_answer: str = "",
        use_api: bool = True, cache_dir: str = "output/society/classified",
    ) -> ReasoningStyleResult:
        return classify_reasoning_style(
            response, question, correct_answer, use_api, cache_dir,
            api_key=self._api_key,
            api_base=self._api_base,
            api_model=self._api_model,
        )

    def classify_error_profile(
        self, response: str, question: str, extracted_answer: str = "",
        correct_answer: str = "", choices: str | list | dict = "",
        dataset_name: str = "", task_type: str = "", subject: str = "",
        use_api: bool = True,
        cache_dir: str = "output/society/classified",
    ) -> ErrorProfileResult:
        return classify_error_profile(
            response=response,
            question=question,
            extracted_answer=extracted_answer,
            correct_answer=correct_answer,
            choices=choices,
            dataset_name=dataset_name,
            task_type=task_type,
            subject=subject,
            use_api=use_api,
            cache_dir=cache_dir,
            api_key=self._api_key,
            api_base=self._api_base,
            api_model=self._api_model,
        )
