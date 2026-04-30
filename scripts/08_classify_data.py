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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
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


def main():
    # Import math answer comparison for robust checking (e.g., "42" == "42.0")
    from src.algorithms.reward import math_answers_equal

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

    for idx, traj in enumerate(trajectories):
        sample_id = traj.get("sample_id", f"sample_{idx}")

        if sample_id in completed_ids:
            continue

        if (idx + 1) % args.batch_size == 0:
            logger.info(f"  Progress: {idx + 1}/{len(trajectories)}")
            # Save checkpoint periodically
            save_checkpoint(output_dir, {
                "completed": list(completed_ids),
                "results": results,
            })

        question = traj["sample"].get("question", "")

        all_responses = iter_trajectory_responses(traj)

        # Classify reasoning style for correct responses
        # and error profile for incorrect responses
        reasoning_styles = []
        error_profiles = []

        # Determine task type for proper answer comparison
        task_type = traj.get("sample", {}).get("task_type", "math")

        # Per-response classifications (preserve agent-level diversity)
        per_response_labels = []

        for resp in all_responses:
            response_text = resp.get("response", "")
            answer = resp.get("answer")
            agent_name = resp.get("agent_name", "")
            response_id = resp.get("response_id", "")
            round_num = resp.get("round", 0)
            agent_id = resp.get("agent_id", 0)
            # Use ground truth from sample, not consensus (consensus may be wrong)
            correct_answer = traj.get("sample", {}).get("answer", "")

            # Use math_answers_equal for math tasks to handle "42" == "42.0"
            if answer:
                if task_type == "math":
                    is_correct = math_answers_equal(answer, correct_answer)
                else:
                    is_correct = answer.upper() == correct_answer.upper()
            else:
                is_correct = False

            per_response_labels.append({
                "sample_id": sample_id,
                "response_id": response_id,
                "round": round_num,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": response_text,
                "answer": answer,
                "is_correct": is_correct,
                "reasoning_style": None,
                "reasoning_style_confidence": 0.0,
                "error_profile": None,
            })

            if is_correct:
                # Correct response - classify reasoning style
                classification_attempts += 1
                try:
                    style, conf = classifier.classify_reasoning_style(question, response_text)
                except Exception as e:
                    logger.warning(f"Reasoning style classification failed for {response_id}: {e}")
                    style, conf = None, 0.0
                if style is None:
                    classification_failures += 1
                reasoning_styles.append((style, conf))
                per_response_labels[-1]["reasoning_style"] = style
                per_response_labels[-1]["reasoning_style_confidence"] = conf
            else:
                # Incorrect response - classify multi-dimensional error profile
                classification_attempts += 1
                try:
                    profile = classifier.classify_error_profile(
                        question, response_text, sample=traj.get("sample", {}),
                        extracted_answer=answer or "",
                    )
                except Exception as e:
                    logger.warning(f"Error profile classification failed for {response_id}: {e}")
                    profile = None
                if profile is None:
                    classification_failures += 1
                if profile:
                    error_profiles.append(profile)
                per_response_labels[-1]["error_profile"] = profile

        # Aggregate classifications at sample level.
        # Filter out None labels (unclassified) from aggregation
        from collections import Counter

        valid_styles = [(s, c) for s, c in reasoning_styles if s is not None]

        reasoning_style = None
        reasoning_confidence = 0.0
        if valid_styles:
            style_counter = Counter([s for s, _ in valid_styles])
            reasoning_style = style_counter.most_common(1)[0][0]
            reasoning_confidence = sum(c for s, c in valid_styles if s == reasoning_style) / len(valid_styles)

        error_profile = None
        if error_profiles:
            dims = ("computation", "reasoning", "knowledge", "grounding", "verification")
            avg_scores = {
                dim: sum(p.get("scores", {}).get(dim, 0.0) for p in error_profiles) / len(error_profiles)
                for dim in dims
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

        result = ClassificationResult(
            sample_id=sample_id,
            reasoning_style=reasoning_style,
            reasoning_style_confidence=reasoning_confidence,
            error_profile=error_profile,
            metadata={
                "num_correct": len(reasoning_styles),
                "num_incorrect": len(error_profiles),
                "num_responses": len(all_responses),
            },
        )

        results.append({
            "sample_id": sample_id,
            "question": question,
            "reasoning_style": reasoning_style,
            "reasoning_style_confidence": reasoning_confidence,
            "error_profile": error_profile,
            "metadata": result.metadata,
            "per_response_labels": per_response_labels,
        })

        completed_ids.add(sample_id)

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
