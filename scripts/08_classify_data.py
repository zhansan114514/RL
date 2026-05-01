"""
Classify bootstrap data by reasoning style and error type.

Uses GLM-4.5 API to classify:
- Reasoning style for correct responses
- Multi-dimensional error profile for incorrect responses

Supports checkpointing for crash recovery.

Usage:
    python scripts/08_classify_data.py \
        --config configs/society/experiment_h100.yaml \
        --api_key YOUR_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.society.data_classifier import (
    ERROR_PROFILE_DIMENSIONS,
    ClassificationError,
    _compute_error_profile_hash,
    _compute_sample_hash,
    _load_cache,
    _save_cache,
    classify_error_profile as shared_classify_error_profile,
    classify_reasoning_style as shared_classify_reasoning_style,
)
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "api_key": "",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "api_model": "glm-4-flash",
    "batch_size": 10,
    "request_timeout": 30,
    "retry_delay": 5,
    "max_retries": 3,
    "api_temperature": 0.1,
    "input_dir": "output/society/bootstrap",
    "output_dir": "output/society/classified",
    "strict_classification": True,
    "max_classification_failure_rate": 0.0,
    "use_batch_api": True,
    "batch_api_max_responses": 20,
    "max_workers": 4,
}


@dataclass
class ClassificationResult:
    """Result of classifying a trajectory."""
    sample_id: str
    reasoning_style: Optional[str]
    reasoning_style_confidence: float
    error_profile: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class GLMClassifier:
    """Classifier using GLM-4.5 API."""

    # Word variants: non-canonical form -> canonical label
    _STYLE_VARIANTS = {
        "algebraic": "algebraic", "algebraically": "algebraic",
        "algebra": "algebraic",
        "direct": "direct", "directly": "direct",
        "backtracking": "backtracking", "backtrack": "backtracking",
    }
    _ERROR_VARIANTS = {
        "computation": "computation", "computational": "computation",
        "calculation": "computation",
        "reasoning": "reasoning",
        "knowledge": "knowledge", "factual": "knowledge",
        "grounding": "grounding", "grounded": "grounding",
        "verification": "verification", "verify": "verification",
        "verifying": "verification",
    }

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 5,
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

    def classify_reasoning_style(
        self,
        question: str,
        response: str,
    ) -> tuple[str | None, float]:
        """Classify reasoning style using GLM API."""
        prompt = f"""Classify the reasoning style in this response into one of:
- algebraic: symbolic manipulation, equations, variables (e.g., "let x =", solving systems)
- direct: direct step-by-step numerical computation without symbolic setup
- backtracking: starts with an attempt, verifies it, then revises if needed

Question: {question}

Response: {response}

Respond with ONLY the style name in lowercase and a confidence score (0-1).
Format: "stylename 0.85"
Example: "algebraic 0.85"
Do NOT include any other text."""

        try:
            return self._call_api(prompt, valid_labels={"algebraic", "direct", "backtracking"})
        except RuntimeError:
            # Secondary retry with minimal prompt
            return self._retry_minimal(
                f"What is the reasoning style? Reply ONE word: algebraic, direct, or backtracking\n\nQuestion: {question}\nResponse: {response[:500]}",
                valid_labels={"algebraic", "direct", "backtracking"},
            )

    def classify_error_profile(
        self,
        question: str,
        response: str,
        sample: Optional[Dict[str, Any]] = None,
        extracted_answer: str = "",
    ) -> Dict[str, Any] | None:
        """Classify error profile using GLM API."""
        sample = sample or {}
        prompt = f"""You are classifying why a model response is wrong.

Dataset/task type: {sample.get("task_type", "unknown")}
Subject/domain: {sample.get("subject", sample.get("category", "unknown"))}

Question: {question}

Choices: {json.dumps(sample.get("choices", ""), ensure_ascii=False)}

Response: {response}

Extracted answer: {extracted_answer}
Correct answer: {sample.get("answer", "")}

Score each error dimension from 0.0 to 1.0:
- computation: numerical calculation, algebra, symbolic manipulation, formula computation
- reasoning: flawed reasoning chain, invalid inference, wrong rule application
- knowledge: wrong factual/domain knowledge, concept confusion
- grounding: ignores or contradicts the question/options, invents unsupported assumptions
- verification: fails to check final answer, option-letter mismatch, self-check failure

Return JSON only with keys: scores, primary, secondary, confidence, evidence."""

        try:
            return self._call_profile_api(prompt)
        except RuntimeError:
            logger.warning("Error profile classification failed after retry, marking as unclassified")
            return None

    def classify_response_batch(
        self,
        question: str,
        sample: Dict[str, Any],
        responses: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Classify all uncached responses for one sample in a single GLM call."""
        payload = []
        for item in responses:
            payload.append({
                "response_id": item.get("response_id", ""),
                "is_correct": bool(item.get("is_correct", False)),
                "extracted_answer": item.get("answer") or "",
                "response": (item.get("response") or "")[:1600],
            })

        prompt = f"""Classify multiple model responses for one dataset sample.

Dataset/task type: {sample.get("task_type", "unknown")}
Subject/domain: {sample.get("subject", sample.get("category", "unknown"))}

Question: {question}

Choices: {json.dumps(sample.get("choices", ""), ensure_ascii=False)}
Correct answer: {sample.get("answer", "")}

Responses:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Rules:
- If is_correct is true, classify reasoning_style as one of: algebraic, direct, backtracking. Set error_profile to null.
- If is_correct is false, classify error_profile. Set reasoning_style to null.
- Error profile scores must include: computation, reasoning, knowledge, grounding, verification, each 0.0 to 1.0.
- Preserve every response_id exactly.

Return JSON only in this exact shape:
{{
  "labels": [
    {{
      "response_id": "same id",
      "reasoning_style": "algebraic|direct|backtracking|null",
      "reasoning_style_confidence": 0.0,
      "error_profile": {{
        "scores": {{
          "computation": 0.0,
          "reasoning": 0.0,
          "knowledge": 0.0,
          "grounding": 0.0,
          "verification": 0.0
        }},
        "primary": "computation|reasoning|knowledge|grounding|verification|unknown",
        "secondary": ["computation|reasoning|knowledge|grounding|verification"],
        "confidence": 0.0,
        "evidence": "short reason"
      }}
    }}
  ]
}}"""
        return self._call_batch_api(prompt)

    def _retry_minimal(self, prompt: str, valid_labels: set[str]) -> tuple[str | None, float]:
        """Secondary retry with a minimal prompt when primary classification fails."""
        try:
            return self._call_api(prompt, valid_labels=valid_labels)
        except RuntimeError:
            logger.warning("Classification failed after retry, marking as unclassified (None)")
            return None, 0.0

    def _call_api(self, prompt: str, valid_labels: set[str] | None = None) -> tuple[str, float]:
        """Call GLM API with retry logic."""
        import requests

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()["choices"][0]["message"]["content"].strip()

                label = self._extract_label(result)
                confidence = self._extract_confidence(result)

                if label and (valid_labels is None or label in valid_labels):
                    return label, confidence

                last_error = f"Unrecognized label in API response: {result!r}"
                logger.warning(
                    f"API returned unrecognized label (attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"API classification failed after {self.max_retries} attempts: {last_error}")

    def _call_profile_api(self, prompt: str) -> Dict[str, Any]:
        """Call GLM API and parse a JSON error profile."""
        import requests

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": 512,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()["choices"][0]["message"]["content"].strip()
                profile = self._extract_profile(result)
                if profile:
                    return profile

                last_error = f"Unrecognized profile JSON in API response: {result!r}"
                logger.warning(
                    f"API returned unrecognized profile (attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"API profile classification failed after {self.max_retries} attempts: {last_error}")

    def _call_batch_api(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """Call GLM API and parse trajectory-level batch classifications."""
        import requests

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": 4096,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()["choices"][0]["message"]["content"].strip()
                labels = self._extract_batch_labels(result)
                if labels:
                    return labels

                last_error = f"Unrecognized batch JSON in API response: {result!r}"
                logger.warning(
                    f"API returned unrecognized batch labels "
                    f"(attempt {attempt + 1}/{self.max_retries}): {last_error}"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Batch API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"API batch classification failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _extract_batch_labels(text: str) -> Dict[str, Dict[str, Any]]:
        """Extract response_id -> classification label from batch JSON."""
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [pos for pos in (start_obj, start_arr) if pos >= 0]
        if starts:
            start = min(starts)
            end = text.rfind("}" if start == start_obj else "]")
            if end >= start:
                text = text[start:end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(
                    r'\\([^"\\/bfnrtu])',
                    lambda m: '\\\\' + m.group(1),
                    text,
                )
                data = json.loads(fixed)
            except json.JSONDecodeError:
                return {}

        if isinstance(data, dict):
            rows = data.get("labels", [])
        elif isinstance(data, list):
            rows = data
        else:
            return {}

        labels: Dict[str, Dict[str, Any]] = {}
        if not isinstance(rows, list):
            return labels
        for row in rows:
            if not isinstance(row, dict):
                continue
            response_id = str(row.get("response_id", "")).strip()
            if response_id:
                labels[response_id] = row
        return labels

    @staticmethod
    def _extract_profile(text: str) -> Dict[str, Any] | None:
        """Extract and normalize a JSON error profile."""
        dims = ("computation", "reasoning", "knowledge", "grounding", "verification")
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
        scores = {}
        for dim in dims:
            try:
                scores[dim] = max(0.0, min(1.0, float(raw_scores.get(dim, 0.0))))
            except (TypeError, ValueError):
                scores[dim] = 0.0

        primary = str(data.get("primary", "")).strip().lower()
        if primary not in dims:
            primary = max(scores.items(), key=lambda kv: kv[1])[0] if any(scores.values()) else "unknown"

        secondary = data.get("secondary", [])
        if isinstance(secondary, str):
            secondary = [secondary]
        if not isinstance(secondary, list):
            secondary = []
        secondary = [
            str(label).strip().lower()
            for label in secondary
            if str(label).strip().lower() in dims and str(label).strip().lower() != primary
        ]

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        return {
            "scores": scores,
            "primary": primary,
            "secondary": secondary,
            "confidence": confidence,
            "evidence": str(data.get("evidence", "")).strip(),
        }

    @classmethod
    def _extract_label(cls, text: str) -> str:
        """Extract a known category label from API response text.

        Handles markdown bold (``**logic**``), word variants ("logical"),
        verbose responses, and punctuation.
        """
        # Strip markdown bold/italic markers, quotes, brackets
        clean = re.sub(r'[*_`"\'\(\)\[\]{}]', ' ', text.lower())
        # Strip "the error type is" / "the response contains a X error" boilerplate

        # 1. Try regex word-boundary match for exact canonical labels
        all_labels = set(cls._STYLE_VARIANTS.values()) | set(cls._ERROR_VARIANTS.values())
        for label in all_labels:
            if re.search(r'\b' + re.escape(label) + r'\b', clean):
                return label

        # 2. Try variant matching (e.g., "logical" -> "logic")
        words = re.findall(r'\b[a-z]+\b', clean)
        for word in words:
            if word in cls._STYLE_VARIANTS:
                return cls._STYLE_VARIANTS[word]
            if word in cls._ERROR_VARIANTS:
                return cls._ERROR_VARIANTS[word]

        return ""

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract a confidence score (0-1 float) from API response text."""
        matches = re.findall(r'(?<!\d)(0?\.\d+|1\.0+)(?!\d)', text)
        if matches:
            try:
                val = float(matches[-1])
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Classify bootstrap data",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="GLM API key (overrides config).",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step02_classify", defaults=STEP_DEFAULTS).to_namespace()

    # CLI override for API key, with env var fallback
    if cli_args.api_key:
        args.api_key = cli_args.api_key
    elif not args.api_key:
        args.api_key = os.environ.get("GLM_API_KEY", "")

    return args


def load_trajectories(input_dir: str) -> List[Dict]:
    """Load trajectories from JSONL file."""
    import glob

    pattern = os.path.join(input_dir, "trajectories.jsonl")
    if not os.path.exists(pattern):
        # Try to find any jsonl file
        files = glob.glob(os.path.join(input_dir, "*.jsonl"))
        if files:
            pattern = files[0]
        else:
            raise FileNotFoundError(f"No trajectories found in {input_dir}")

    trajectories = []
    with open(pattern) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    logger.info(f"Loaded {len(trajectories)} trajectories from {pattern}")
    return trajectories


def load_checkpoint(output_dir: str) -> Dict[str, Any]:
    """Load existing checkpoint if exists."""
    checkpoint_file = os.path.join(output_dir, "classification_checkpoint.json")

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            return json.load(f)

    return {"completed": [], "results": []}


def save_checkpoint(output_dir: str, checkpoint_data: Dict[str, Any]):
    """Save checkpoint for crash recovery."""
    checkpoint_file = os.path.join(output_dir, "classification_checkpoint.json")

    os.makedirs(output_dir, exist_ok=True)

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def make_response_id(sample_id: str, round_num: int, agent_id: int) -> str:
    return f"{sample_id}_round_{round_num}_agent_{agent_id}"


def iter_trajectory_responses(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return every bootstrap response with a stable response_id."""
    sample_id = traj.get("sample_id", "sample")
    responses: List[Dict[str, Any]] = []

    for resp in traj.get("initial_responses", []):
        item = dict(resp)
        round_num = int(item.get("round", 0))
        agent_id = int(item.get("agent_id", 0))
        item["sample_id"] = sample_id
        item["round"] = round_num
        item["agent_id"] = agent_id
        item["response_id"] = item.get("response_id") or make_response_id(
            sample_id, round_num, agent_id,
        )
        responses.append(item)

    for round_responses in traj.get("debate_rounds", []):
        for resp in round_responses:
            item = dict(resp)
            round_num = int(item.get("round", 0))
            agent_id = int(item.get("agent_id", 0))
            item["sample_id"] = sample_id
            item["round"] = round_num
            item["agent_id"] = agent_id
            item["response_id"] = item.get("response_id") or make_response_id(
                sample_id, round_num, agent_id,
            )
            responses.append(item)

    return responses


def normalize_profile(profile: Any) -> Optional[Dict[str, Any]]:
    """Normalize an error profile dict from either batch or single-response API."""
    if not isinstance(profile, dict):
        return None

    raw_scores = profile.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores = {}
    for dim in ERROR_PROFILE_DIMENSIONS:
        try:
            scores[dim] = max(0.0, min(1.0, float(raw_scores.get(dim, 0.0))))
        except (TypeError, ValueError):
            scores[dim] = 0.0

    primary = str(profile.get("primary", "")).strip().lower()
    if primary not in ERROR_PROFILE_DIMENSIONS:
        primary = max(scores.items(), key=lambda kv: kv[1])[0] if any(scores.values()) else "unknown"

    secondary_raw = profile.get("secondary", [])
    if isinstance(secondary_raw, str):
        secondary_raw = [secondary_raw]
    if not isinstance(secondary_raw, list):
        secondary_raw = []
    secondary = []
    for label in secondary_raw:
        label = str(label).strip().lower()
        if label in ERROR_PROFILE_DIMENSIONS and label != primary and label not in secondary:
            secondary.append(label)

    try:
        confidence = max(0.0, min(1.0, float(profile.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "scores": scores,
        "primary": primary,
        "secondary": secondary[:2],
        "confidence": confidence,
        "evidence": str(profile.get("evidence", "")).strip(),
    }


def style_cache_path(output_dir: str, question: str, response: str) -> Path:
    return Path(output_dir) / f"style_{_compute_sample_hash(question, response)}.json"


def error_cache_path(
    output_dir: str,
    question: str,
    response: str,
    sample: Dict[str, Any],
) -> Path:
    choices = sample.get("choices", "")
    choices_text = json.dumps(choices, ensure_ascii=False) if isinstance(choices, (list, dict)) else str(choices or "")
    return Path(output_dir) / f"error_profile_{_compute_error_profile_hash(
        question=question,
        response=response,
        correct_answer=sample.get("answer", ""),
        choices=choices_text,
        dataset_name=sample.get("dataset_name", ""),
        task_type=sample.get("task_type", ""),
        subject=sample.get("subject", sample.get("category", "")),
    )}.json"


def load_cached_label(
    output_dir: str,
    question: str,
    sample: Dict[str, Any],
    label: Dict[str, Any],
) -> bool:
    """Populate label from shared cache. Returns True on cache hit."""
    response = label.get("response", "")
    if label.get("is_correct"):
        cached = _load_cache(style_cache_path(output_dir, question, response))
        if not cached:
            return False
        style = str(cached.get("style", "")).strip().lower()
        if style not in {"algebraic", "direct", "backtracking"}:
            return False
        label["reasoning_style"] = style
        label["reasoning_style_confidence"] = float(cached.get("confidence", 0.9))
        return True

    cached = _load_cache(error_cache_path(output_dir, question, response, sample))
    if not cached:
        return False
    profile = normalize_profile(cached)
    if profile is None:
        return False
    label["error_profile"] = profile
    return True


def save_label_cache(
    output_dir: str,
    question: str,
    sample: Dict[str, Any],
    label: Dict[str, Any],
) -> None:
    """Write a successful label to the shared cache used by Phase 4/5."""
    response = label.get("response", "")
    if label.get("is_correct") and label.get("reasoning_style"):
        _save_cache(style_cache_path(output_dir, question, response), {
            "style": label["reasoning_style"],
            "confidence": label.get("reasoning_style_confidence", 0.9),
            "raw_response": "batch_api",
        })
    elif not label.get("is_correct") and label.get("error_profile"):
        profile = label["error_profile"]
        _save_cache(error_cache_path(output_dir, question, response, sample), {
            "scores": profile.get("scores", {}),
            "primary": profile.get("primary", "unknown"),
            "secondary": profile.get("secondary", []),
            "confidence": profile.get("confidence", 0.0),
            "evidence": profile.get("evidence", ""),
            "raw_response": "batch_api",
        })


def build_per_response_labels(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create response labels with deterministic correctness before GLM calls."""
    from src.algorithms.reward import math_answers_equal

    sample_id = traj.get("sample_id", "sample")
    sample = traj.get("sample", {})
    task_type = sample.get("task_type", "math")
    correct_answer = sample.get("answer", "")
    labels = []

    for resp in iter_trajectory_responses(traj):
        response_text = resp.get("response", "")
        answer = resp.get("answer")
        if answer:
            if task_type == "math":
                is_correct = math_answers_equal(answer, correct_answer)
            else:
                is_correct = str(answer).upper() == str(correct_answer).upper()
        else:
            is_correct = False

        labels.append({
            "sample_id": sample_id,
            "response_id": resp.get("response_id", ""),
            "round": resp.get("round", 0),
            "agent_id": resp.get("agent_id", 0),
            "agent_name": resp.get("agent_name", ""),
            "response": response_text,
            "answer": answer,
            "is_correct": is_correct,
            "reasoning_style": None,
            "reasoning_style_confidence": 0.0,
            "error_profile": None,
            "classification_source": None,
        })
    return labels


def apply_batch_row(label: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """Apply and validate one batch classification row."""
    if label.get("is_correct"):
        style = row.get("reasoning_style")
        style = str(style).strip().lower() if style is not None else ""
        if style not in {"algebraic", "direct", "backtracking"}:
            return False
        try:
            confidence = max(0.0, min(1.0, float(row.get("reasoning_style_confidence", 0.9))))
        except (TypeError, ValueError):
            confidence = 0.9
        label["reasoning_style"] = style
        label["reasoning_style_confidence"] = confidence
        return True

    profile = normalize_profile(row.get("error_profile"))
    if profile is None:
        return False
    label["error_profile"] = profile
    return True


def classify_label_single(
    classifier: GLMClassifier,
    label: Dict[str, Any],
    question: str,
    sample: Dict[str, Any],
    output_dir: str,
) -> bool:
    """Classify one label through the shared single-response classifier."""
    try:
        if label.get("is_correct"):
            result = shared_classify_reasoning_style(
                response=label.get("response", ""),
                question=question,
                correct_answer=sample.get("answer", ""),
                use_api=True,
                cache_dir=output_dir,
                api_key=classifier.api_key,
                api_base=classifier.api_base,
                api_model=classifier.model,
            )
            label["reasoning_style"] = result.style.value
            label["reasoning_style_confidence"] = result.confidence
        else:
            result = shared_classify_error_profile(
                response=label.get("response", ""),
                question=question,
                extracted_answer=label.get("answer") or "",
                correct_answer=sample.get("answer", ""),
                choices=sample.get("choices", ""),
                dataset_name=sample.get("dataset_name", ""),
                task_type=sample.get("task_type", ""),
                subject=sample.get("subject", sample.get("category", "")),
                use_api=True,
                cache_dir=output_dir,
                api_key=classifier.api_key,
                api_base=classifier.api_base,
                api_model=classifier.model,
            )
            label["error_profile"] = {
                "scores": result.scores,
                "primary": result.primary,
                "secondary": result.secondary,
                "confidence": result.confidence,
                "evidence": result.evidence,
            }
        label["classification_source"] = "single_api"
        return True
    except ClassificationError as e:
        logger.warning(f"Single classification failed for {label.get('response_id', '')}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected single classification failure for {label.get('response_id', '')}: {e}")
    return False


def aggregate_result(
    sample_id: str,
    question: str,
    all_responses_count: int,
    per_response_labels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-response labels into the sample-level summary."""
    reasoning_styles = [
        (label.get("reasoning_style"), label.get("reasoning_style_confidence", 0.0))
        for label in per_response_labels
        if label.get("is_correct") and label.get("reasoning_style")
    ]
    error_profiles = [
        label.get("error_profile")
        for label in per_response_labels
        if not label.get("is_correct") and label.get("error_profile")
    ]

    reasoning_style = None
    reasoning_confidence = 0.0
    if reasoning_styles:
        style_counter = Counter([s for s, _ in reasoning_styles])
        reasoning_style = style_counter.most_common(1)[0][0]
        reasoning_confidence = sum(
            c for s, c in reasoning_styles if s == reasoning_style
        ) / len(reasoning_styles)

    error_profile = None
    if error_profiles:
        avg_scores = {
            dim: sum(p.get("scores", {}).get(dim, 0.0) for p in error_profiles) / len(error_profiles)
            for dim in ERROR_PROFILE_DIMENSIONS
        }
        primary_counter = Counter([p.get("primary", "unknown") for p in error_profiles])
        primary = primary_counter.most_common(1)[0][0]
        error_profile = {
            "scores": avg_scores,
            "primary": primary,
            "secondary": [
                dim for dim, _ in sorted(avg_scores.items(), key=lambda kv: kv[1], reverse=True)
                if dim != primary
            ][:2],
            "confidence": sum(p.get("confidence", 0.0) for p in error_profiles) / len(error_profiles),
            "evidence": "sample-level aggregate",
        }

    metadata = {
        "num_correct": len(reasoning_styles),
        "num_incorrect": len(error_profiles),
        "num_responses": all_responses_count,
        "classification_sources": dict(Counter(
            label.get("classification_source") or "unclassified"
            for label in per_response_labels
        )),
    }
    return {
        "sample_id": sample_id,
        "question": question,
        "reasoning_style": reasoning_style,
        "reasoning_style_confidence": reasoning_confidence,
        "error_profile": error_profile,
        "metadata": metadata,
        "per_response_labels": per_response_labels,
    }


def classify_trajectory(
    traj: Dict[str, Any],
    idx: int,
    classifier: GLMClassifier,
    args: Any,
) -> Dict[str, Any]:
    """Classify one trajectory with cache, batch API, and single-call fallback."""
    sample_id = traj.get("sample_id", f"sample_{idx}")
    sample = dict(traj.get("sample", {}))
    sample.setdefault("dataset_name", getattr(args, "dataset", ""))
    question = sample.get("question", "")
    labels = build_per_response_labels(traj)

    attempts = 0
    failures = 0
    cache_hits = 0
    batch_calls = 0
    single_calls = 0

    pending = []
    for label in labels:
        attempts += 1
        if load_cached_label(args.output_dir, question, sample, label):
            label["classification_source"] = "cache"
            cache_hits += 1
        else:
            pending.append(label)

    use_batch = bool(getattr(args, "use_batch_api", True))
    max_batch = max(1, int(getattr(args, "batch_api_max_responses", 20)))
    if use_batch and pending:
        for start in range(0, len(pending), max_batch):
            chunk = pending[start:start + max_batch]
            batch_calls += 1
            try:
                batch_labels = classifier.classify_response_batch(question, sample, chunk)
            except Exception as e:
                logger.warning(f"Batch classification failed for {sample_id}; falling back to single calls: {e}")
                batch_labels = {}

            for label in chunk:
                row = batch_labels.get(label.get("response_id", ""))
                if row and apply_batch_row(label, row):
                    label["classification_source"] = "batch_api"
                    save_label_cache(args.output_dir, question, sample, label)
                else:
                    if classify_label_single(classifier, label, question, sample, args.output_dir):
                        single_calls += 1
                    else:
                        failures += 1
    else:
        for label in pending:
            if classify_label_single(classifier, label, question, sample, args.output_dir):
                single_calls += 1
            else:
                failures += 1

    result = aggregate_result(
        sample_id=sample_id,
        question=question,
        all_responses_count=len(labels),
        per_response_labels=labels,
    )
    return {
        "idx": idx,
        "sample_id": sample_id,
        "result": result,
        "attempts": attempts,
        "failures": failures,
        "cache_hits": cache_hits,
        "batch_calls": batch_calls,
        "single_calls": single_calls,
    }


def main():
    args = parse_args()

    # Setup directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Classify Bootstrap Data")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  API model: {args.api_model}")
    logger.info(
        "  GLM optimization: "
        f"use_batch_api={getattr(args, 'use_batch_api', True)}, "
        f"batch_api_max_responses={getattr(args, 'batch_api_max_responses', 20)}, "
        f"max_workers={getattr(args, 'max_workers', 4)}"
    )
    logger.info("=" * 60)

    # Load trajectories
    logger.info("[Step 1] Loading trajectories...")
    trajectories = load_trajectories(input_dir)

    # Load checkpoint
    logger.info("[Step 2] Loading checkpoint...")
    checkpoint = load_checkpoint(output_dir)
    completed_ids = set(checkpoint["completed"])
    results = checkpoint["results"]
    classification_attempts = 0
    classification_failures = 0
    cache_hits = 0
    batch_api_calls = 0
    single_api_calls = 0

    logger.info(f"  Already classified: {len(completed_ids)}/{len(trajectories)}")

    # Initialize classifier
    logger.info("[Step 3] Initializing classifier...")

    if not args.api_key:
        logger.error("API key is required. Set via --api_key or GLM_API_KEY environment variable.")
        sys.exit(1)

    classifier = GLMClassifier(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.api_model,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        temperature=args.api_temperature,
    )

    # Classify trajectories
    logger.info("[Step 4] Classifying trajectories...")

    pending_items = [
        (idx, traj)
        for idx, traj in enumerate(trajectories)
        if traj.get("sample_id", f"sample_{idx}") not in completed_ids
    ]
    max_workers = max(1, int(getattr(args, "max_workers", 4)))
    if max_workers == 1:
        futures = []
        for idx, traj in pending_items:
            futures.append(classify_trajectory(traj, idx, classifier, args))
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(classify_trajectory, traj, idx, classifier, args): idx
                for idx, traj in pending_items
            }
            for future in as_completed(future_map):
                futures.append(future.result())

    for processed, item in enumerate(sorted(futures, key=lambda x: x["idx"]), start=1):
        results.append(item["result"])
        completed_ids.add(item["sample_id"])
        classification_attempts += item["attempts"]
        classification_failures += item["failures"]
        cache_hits += item["cache_hits"]
        batch_api_calls += item["batch_calls"]
        single_api_calls += item["single_calls"]

        if processed % args.batch_size == 0:
            logger.info(
                f"  Progress: {len(completed_ids)}/{len(trajectories)} "
                f"(cache_hits={cache_hits}, batch_api_calls={batch_api_calls}, "
                f"single_api_calls={single_api_calls}, failures={classification_failures})"
            )
            save_checkpoint(output_dir, {
                "completed": list(completed_ids),
                "results": results,
            })

    save_checkpoint(output_dir, {
        "completed": list(completed_ids),
        "results": results,
    })

    failure_rate = (
        classification_failures / classification_attempts
        if classification_attempts
        else 0.0
    )
    strict_classification = bool(getattr(args, "strict_classification", True))
    max_failure_rate = float(getattr(args, "max_classification_failure_rate", 0.0))
    if strict_classification and failure_rate > max_failure_rate:
        raise RuntimeError(
            f"Classification failure rate {failure_rate:.3f} exceeds threshold "
            f"{max_failure_rate:.3f} ({classification_failures}/{classification_attempts})"
        )

    # Save final results
    logger.info("[Step 5] Saving results...")

    output_file = os.path.join(output_dir, "classified_data.json")
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "metadata": {
                "total_trajectories": len(trajectories),
                "api_model": args.api_model,
                "strict_classification": strict_classification,
                "classification_attempts": classification_attempts,
                "classification_failures": classification_failures,
                "classification_failure_rate": failure_rate,
                "max_classification_failure_rate": max_failure_rate,
                "cache_hits": cache_hits,
                "batch_api_calls": batch_api_calls,
                "single_api_calls": single_api_calls,
                "use_batch_api": bool(getattr(args, "use_batch_api", True)),
                "batch_api_max_responses": int(getattr(args, "batch_api_max_responses", 20)),
                "max_workers": max_workers,
            },
        }, f, indent=2, ensure_ascii=False)

    # Save per-style splits
    logger.info("[Step 6] Creating per-style splits...")

    style_splits = {}
    profile_splits = {}

    # Per-response granular splits
    per_response_style_splits = {}
    per_response_profile_splits = {}

    for result in results:
        style = result["reasoning_style"]
        profile = result.get("error_profile")
        sample_id = result["sample_id"]

        if style:
            style_splits.setdefault(style, []).append(sample_id)

        if profile and profile.get("primary") and profile["primary"] != "unknown":
            profile_splits.setdefault(profile["primary"], []).append(sample_id)

        # Build per-response splits for finer-grained data diversification
        per_labels = result.get("per_response_labels", [])
        for label in per_labels:
            if label.get("reasoning_style"):
                per_response_style_splits.setdefault(
                    label["reasoning_style"], [],
                ).append({
                    "sample_id": sample_id,
                    "response_id": label.get("response_id", ""),
                    "round": label.get("round", 0),
                    "agent_id": label.get("agent_id", 0),
                    "agent_name": label.get("agent_name", ""),
                    "is_correct": label.get("is_correct", False),
                })
            profile = label.get("error_profile")
            if profile and profile.get("primary") and profile["primary"] != "unknown":
                per_response_profile_splits.setdefault(
                    profile["primary"], [],
                ).append({
                    "sample_id": sample_id,
                    "response_id": label.get("response_id", ""),
                    "round": label.get("round", 0),
                    "agent_id": label.get("agent_id", 0),
                    "agent_name": label.get("agent_name", ""),
                    "is_correct": label.get("is_correct", False),
                    "scores": profile.get("scores", {}),
                    "confidence": profile.get("confidence", 0.0),
                })

    splits_file = os.path.join(output_dir, "splits.json")
    with open(splits_file, "w") as f:
        json.dump({
            "reasoning_styles": style_splits,
            "error_profiles": profile_splits,
        }, f, indent=2)

    # Save per-response splits for finer-grained Actor/Critic diversification
    per_response_splits_file = os.path.join(output_dir, "per_response_splits.json")
    with open(per_response_splits_file, "w") as f:
        json.dump({
            "reasoning_styles": per_response_style_splits,
            "error_profiles": per_response_profile_splits,
        }, f, indent=2)

    logger.info(f"  Reasoning style splits: {list(style_splits.keys())}")
    logger.info(f"  Error profile splits: {list(profile_splits.keys())}")
    logger.info(f"  Per-response style splits: { {k: len(v) for k, v in per_response_style_splits.items()} }")
    logger.info(f"  Per-response profile splits: { {k: len(v) for k, v in per_response_profile_splits.items()} }")

    logger.info("=" * 60)
    logger.info("Classification complete!")
    logger.info(f"  Results: {output_file}")
    logger.info(f"  Splits: {splits_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
