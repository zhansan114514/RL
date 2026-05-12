"""
Data classifier for categorizing reasoning styles and critic error profiles.

Uses an OpenAI-compatible chat-completions API for classification with local caching.
Raises ClassificationError when API is required but unavailable,
instead of silently falling back to unreliable heuristics.

From experiment plan:
- Reasoning styles (for correct responses): direct/evidence/elimination
- Error profiles (for incorrect responses): reasoning/knowledge/grounding/verification
  plus format_failure/ambiguous routing labels
"""

from __future__ import annotations

import json
import logging
import os
import time
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
    primary_style: ReasoningStyle
    secondary_styles: list[ReasoningStyle]
    format_status: str
    confidence: float
    evidence: str = ""
    raw_response: str = ""

    @property
    def style(self) -> ReasoningStyle:
        """Alias for callers that need the primary style enum."""
        return self.primary_style


@dataclass
class ErrorProfileResult:
    """Result of multi-dimensional error profile classification."""
    format_status: str
    scores: dict[str, float]
    primary: str
    secondary: list[str]
    confidence: float
    evidence: str = ""
    raw_response: str = ""


# ============================================================
# API configuration (from experiment plan)
# ============================================================

DEFAULT_API_URL = "https://api.labforge.top/v1/chat/completions"
DEFAULT_API_KEY = (
    os.environ.get("GPT_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("GLM_API_KEY", "")
)
DEFAULT_API_MODEL = "gpt5.5"
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 5.0


def normalize_chat_api_url(api_url: str) -> str:
    """Return a concrete OpenAI-compatible chat completions endpoint."""
    url = (api_url or DEFAULT_API_URL).strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"

# Classification prompts
REASONING_STYLE_PROMPT = """Classify the reasoning style of this model response for an MMLU-style task.

Question:
{question}

Response:
{response}

Ignore the final answer sentence and classify the reasoning body.

Definitions:
- direct: concise answer with minimal explanation; does not systematically cite evidence or compare options
- evidence: uses facts, definitions, concepts, question wording, or domain knowledge as explicit evidence
- elimination: compares answer options, rules out alternatives, or explains why one option is better than the others

Return JSON only (pick ONE value for each field, do NOT use "|"):
{{
  "primary_style": "one of: direct, evidence, elimination",
  "secondary_styles": ["subset of: direct, evidence, elimination"],
  "format_status": "one of: valid, answer_only, empty_or_invalid",
  "confidence": 0.0,
  "evidence": "short reason"
}}"""

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

- reasoning: flawed reasoning chain, invalid inference, wrong rule application
- knowledge: wrong factual/domain knowledge, concept confusion
- grounding: ignores or contradicts the question/options, invents unsupported assumptions
- verification: fails to check final answer, option-letter mismatch, self-check failure

Return JSON only (pick ONE value for each field, do NOT use "|"):
{{
  "format_status": "one of: valid, answer_only, empty_or_invalid",
  "scores": {{
    "reasoning": float,
    "knowledge": float,
    "grounding": float,
    "verification": float
  }},
  "primary": "one of: reasoning, knowledge, grounding, verification, format_failure, ambiguous",
  "secondary": ["subset of: reasoning, knowledge, grounding, verification"],
  "confidence": float,
  "evidence": "short reason"
}}
"""

ERROR_PROFILE_DIMENSIONS = (
    "reasoning",
    "knowledge",
    "grounding",
    "verification",
)
ERROR_PROFILE_PRIMARY_LABELS = ERROR_PROFILE_DIMENSIONS + ("format_failure", "ambiguous")
FORMAT_STATUSES = {"valid", "answer_only", "empty_or_invalid"}


# ============================================================
# API availability check
# ============================================================

def _api_key_help() -> str:
    return "api_key, GPT_API_KEY, OPENAI_API_KEY, or GLM_API_KEY"


def check_api_available(
    api_url: str = DEFAULT_API_URL,
    api_key: str | None = None,
) -> tuple[bool, str]:
    """Check if the configured chat-completions API is reachable.

    Returns:
        (available, reason) tuple.  `available` is True when the API
        can be used for classification.  `reason` describes the problem
        when unavailable.
    """
    if api_key is None:
        api_key = DEFAULT_API_KEY
    if not api_key:
        return False, f"API key is not set ({_api_key_help()})"
    try:
        import requests
    except ImportError:
        return False, "requests package is not installed"

    api_url = normalize_chat_api_url(api_url)

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
    max_tokens: int = 256,
    request_timeout: int | float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int | float = DEFAULT_RETRY_DELAY,
) -> str:
    """Call GLM API for classification with retries.

    Raises ClassificationError on any failure instead of silently
    returning None.
    """
    if not api_key:
        raise ClassificationError(
            f"API key not set. Configure {_api_key_help()}."
        )
    try:
        import requests
    except ImportError:
        raise ClassificationError(
            "requests package not installed. Install via: pip install requests"
        )

    api_url = normalize_chat_api_url(api_url)
    attempts = max(1, int(max_retries))
    base_delay = max(0.0, float(retry_delay))
    timeout = max(1.0, float(request_timeout))
    last_error = ""

    def _retryable_http_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code == 429 or (status_code is not None and 500 <= status_code < 600)

    for attempt in range(attempts):
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
                timeout=timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if content is None:
                raise ClassificationError("API returned null content (possible content filter)")
            result = content.strip()
            return result
        except requests.exceptions.Timeout as e:
            last_error = f"API request timed out: {e}"
            retryable = True
        except requests.exceptions.ConnectionError as e:
            last_error = f"Cannot connect to API: {e}"
            retryable = True
        except requests.exceptions.HTTPError as e:
            last_error = f"API HTTP error: {e}"
            retryable = _retryable_http_error(e)
            if not retryable:
                raise ClassificationError(last_error) from e
        except ClassificationError:
            raise
        except (KeyError, IndexError, AttributeError) as e:
            raise ClassificationError(f"Unexpected API response format: {e}") from e
        except Exception as e:
            raise ClassificationError(f"API call failed: {e}") from e

        if not retryable or attempt == attempts - 1:
            break

        delay = base_delay * (2 ** attempt)
        logger.warning(
            "API classification call failed "
            f"(attempt {attempt + 1}/{attempts}); retrying in {delay:.1f}s: {last_error}"
        )
        if delay > 0:
            time.sleep(delay)

    raise ClassificationError(
        f"API classification failed after {attempts} attempts: {last_error}"
    )


# ============================================================
# Response parsers
# ============================================================

def _parse_style_response(response: str) -> Optional[ReasoningStyle]:
    """Parse a style label from JSON or fallback text."""
    if not response:
        return None
    import re

    data = _extract_json(response)
    if isinstance(data, dict):
        label = str(data.get("primary_style", "")).strip().lower()
        # Handle pipe-separated multi-value strings (e.g. "direct|evidence|elimination")
        label = label.split("|")[0].strip()
        try:
            return ReasoningStyle(label)
        except ValueError:
            pass

    clean = re.sub(r'[*_`"\'\(\)\[\]{}]', ' ', response.lower().strip())
    # Count occurrences of each style rather than matching first-hit;
    # this prevents "direct" always winning when multiple styles appear
    # in a truncated multi-value string.
    style_counts: dict[ReasoningStyle, int] = {}
    for style in ReasoningStyle:
        pattern = r'\b' + re.escape(style.value) + r'\b'
        style_counts[style] = len(re.findall(pattern, clean))

    # If only one style was found, return it directly
    found = [s for s, c in style_counts.items() if c > 0]
    if len(found) == 1:
        return found[0]
    # If zero or multiple, fall through to variant matching

    variants = {
        "directly": ReasoningStyle.DIRECT,
        "concise": ReasoningStyle.DIRECT,
        "minimal": ReasoningStyle.DIRECT,
        "evidence_based": ReasoningStyle.EVIDENCE,
        "evidence-based": ReasoningStyle.EVIDENCE,
        "facts": ReasoningStyle.EVIDENCE,
        "factual": ReasoningStyle.EVIDENCE,
        "definition": ReasoningStyle.EVIDENCE,
        "concept": ReasoningStyle.EVIDENCE,
        "domain_knowledge": ReasoningStyle.EVIDENCE,
        "option_elimination": ReasoningStyle.ELIMINATION,
        "option-elimination": ReasoningStyle.ELIMINATION,
        "eliminate": ReasoningStyle.ELIMINATION,
        "eliminating": ReasoningStyle.ELIMINATION,
        "compare": ReasoningStyle.ELIMINATION,
        "comparison": ReasoningStyle.ELIMINATION,
        "rule_out": ReasoningStyle.ELIMINATION,
    }
    words = re.findall(r'\b[a-z_]+\b', clean)
    for word in words:
        if word in variants:
            return variants[word]
    return None


def _parse_style_json_response(response: str) -> Optional[ReasoningStyleResult]:
    """Parse the JSON style-classification response."""
    data = _extract_json(response)
    if not isinstance(data, dict):
        style = _parse_style_response(response)
        if style is None:
            return None
        return ReasoningStyleResult(
            primary_style=style,
            secondary_styles=[],
            format_status="valid",
            confidence=0.9,
            evidence="parsed fallback label",
            raw_response=response or "",
        )

    style = _parse_style_response(response)
    if style is None:
        return None

    secondary_raw = data.get("secondary_styles", [])
    if isinstance(secondary_raw, str):
        secondary_raw = [secondary_raw]
    secondary_styles: list[ReasoningStyle] = []
    if isinstance(secondary_raw, list):
        for label in secondary_raw:
            try:
                parsed = ReasoningStyle(str(label).strip().lower())
            except ValueError:
                continue
            if parsed != style and parsed not in secondary_styles:
                secondary_styles.append(parsed)

    format_status = str(data.get("format_status", "valid")).strip().lower()
    if format_status not in FORMAT_STATUSES:
        format_status = "valid"

    return ReasoningStyleResult(
        primary_style=style,
        secondary_styles=secondary_styles,
        format_status=format_status,
        confidence=_clamp01(data.get("confidence"), 0.5),
        evidence=str(data.get("evidence", "")).strip(),
        raw_response=response or "",
    )


def _extract_json(response: str) -> Optional[object]:
    """Extract a JSON object or array from model output."""
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
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            fixed = re.sub(
                r'\\([^"\\/bfnrtu])',
                lambda m: '\\\\' + m.group(1),
                text,
            )
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _clamp01(value: object, default: float = 0.0) -> float:
    """Convert a value to a bounded 0..1 float."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _ambiguous_error_profile(raw_response: str = "", evidence: str = "") -> ErrorProfileResult:
    """Return an explicit ambiguous profile without assigning a specialist."""
    return ErrorProfileResult(
        format_status="empty_or_invalid",
        scores={dim: 0.0 for dim in ERROR_PROFILE_DIMENSIONS},
        primary="ambiguous",
        secondary=[],
        confidence=0.0,
        evidence=evidence,
        raw_response=raw_response,
    )


def _parse_error_profile_response(response: str) -> Optional[ErrorProfileResult]:
    """Parse a JSON error-profile API response."""
    if not response:
        return None

    data = _extract_json(response)
    if not isinstance(data, dict):
        return None

    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {
        dim: _clamp01(raw_scores.get(dim), 0.0)
        for dim in ERROR_PROFILE_DIMENSIONS
    }

    primary = str(data.get("primary", "")).strip().lower()
    # Handle pipe-separated multi-value strings
    primary = primary.split("|")[0].strip()
    if primary not in ERROR_PROFILE_PRIMARY_LABELS:
        primary = max(scores.items(), key=lambda kv: kv[1])[0] if any(scores.values()) else "ambiguous"

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
    format_status = str(data.get("format_status", "valid")).strip().lower()
    if format_status not in FORMAT_STATUSES:
        format_status = "valid"
    if format_status in {"answer_only", "empty_or_invalid"}:
        primary = "format_failure"
        secondary = [
            label
            for label in secondary
            if label in ERROR_PROFILE_DIMENSIONS
        ]

    return ErrorProfileResult(
        format_status=format_status,
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
    request_timeout: int | float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int | float = DEFAULT_RETRY_DELAY,
) -> ReasoningStyleResult:
    """
    Classify the reasoning style of a correct response.

    Uses GLM API with local caching.
    Raises ClassificationError if use_api=True but API is unavailable
    and no cached result exists.
    """
    sample_hash = _compute_sample_hash(question, response)
    cache_path = Path(cache_dir) / f"style_v4_mmlu_native3_{sample_hash}.json"

    # Check cache first (always succeeds regardless of API)
    cached = _load_cache(cache_path)
    if cached:
        return ReasoningStyleResult(
            primary_style=ReasoningStyle(cached.get("primary_style") or cached["style"]),
            secondary_styles=[
                ReasoningStyle(s)
                for s in cached.get("secondary_styles", [])
                if s in {style.value for style in ReasoningStyle}
            ],
            format_status=cached.get("format_status", "valid"),
            confidence=cached.get("confidence", 0.9),
            evidence=cached.get("evidence", ""),
            raw_response=cached.get("raw_response", ""),
        )

    # API classification
    if use_api:
        prompt = REASONING_STYLE_PROMPT.format(
            question=question,
            response=response[:2000],
        )
        api_response = _call_api(
            prompt,
            api_url=api_base or DEFAULT_API_URL,
            api_key=api_key or DEFAULT_API_KEY,
            model=api_model or DEFAULT_API_MODEL,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        result = _parse_style_json_response(api_response)

        if result is not None:
            _save_cache(cache_path, {
                "style": result.primary_style.value,
                "primary_style": result.primary_style.value,
                "secondary_styles": [s.value for s in result.secondary_styles],
                "format_status": result.format_status,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "raw_response": api_response or "",
            })
            return result

        # API responded but we couldn't parse it
        raise ClassificationError(
            f"Could not parse reasoning style from API response: {api_response!r}"
        )

    raise ClassificationError("Classification requires API or cached result.")


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
    request_timeout: int | float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int | float = DEFAULT_RETRY_DELAY,
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
        subject=f"{subject}|dims={','.join(ERROR_PROFILE_DIMENSIONS)}",
    )
    cache_path = Path(cache_dir) / f"error_profile_{sample_hash}.json"

    # Check cache first
    cached = _load_cache(cache_path)
    if cached:
        cached_format_status = cached.get("format_status", "valid")
        cached_primary = cached.get("primary", "ambiguous")
        if cached_format_status in {"answer_only", "empty_or_invalid"}:
            cached_primary = "format_failure"
        return ErrorProfileResult(
            format_status=cached_format_status,
            scores={dim: _clamp01(cached.get("scores", {}).get(dim), 0.0)
                    for dim in ERROR_PROFILE_DIMENSIONS},
            primary=cached_primary,
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
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        profile = _parse_error_profile_response(api_response)

        if profile is not None:
            _save_cache(cache_path, {
                "format_status": profile.format_status,
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

    raise ClassificationError("Classification requires API or cached result.")


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
        request_timeout: int | float = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: int | float = DEFAULT_RETRY_DELAY,
    ):
        self._api_key = api_key
        self._api_base = api_base
        self._api_model = api_model
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def classify_reasoning_style(
        self, response: str, question: str, correct_answer: str = "",
        use_api: bool = True, cache_dir: str = "output/society/classified",
    ) -> ReasoningStyleResult:
        return classify_reasoning_style(
            response, question, correct_answer, use_api, cache_dir,
            api_key=self._api_key,
            api_base=self._api_base,
            api_model=self._api_model,
            request_timeout=self._request_timeout,
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
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
            request_timeout=self._request_timeout,
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )
