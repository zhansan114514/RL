"""
Classify bootstrap data by reasoning style and error type.

Uses GLM-4.5 API to classify:
- Reasoning style for correct responses
- Error type for incorrect responses

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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "input_dir", "output_dir")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "input_dir": "output/society/bootstrap",
    "output_dir": "output/society/classified",
    "api_key": "",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "api_model": "glm-4-flash",
    "batch_size": 10,
    "request_timeout": 30,
    "retry_delay": 5,
    "max_retries": 3,
    "use_cache": True,
}


@dataclass
class ClassificationResult:
    """Result of classifying a trajectory."""
    sample_id: str
    reasoning_style: Optional[str]
    reasoning_style_confidence: float
    error_type: Optional[str]
    error_type_confidence: float
    metadata: Dict[str, Any]


class GLMClassifier:
    """Classifier using GLM-4.5 API."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # Valid fallback values per classification type
    _STYLE_FALLBACK = "direct"
    _ERROR_FALLBACK = "logic"

    def classify_reasoning_style(
        self,
        question: str,
        response: str,
    ) -> tuple[str, float]:
        """Classify reasoning style using GLM API."""
        prompt = f"""Classify the reasoning style in this response into one of:
- algebraic: symbolic manipulation, equations, variables (e.g., "let x =", solving systems)
- direct: direct step-by-step numerical computation without symbolic setup
- backtracking: starts with an attempt, verifies it, then revises if needed

Question: {question}

Response: {response}

Respond with just the style name and a confidence score (0-1), e.g., "algebraic 0.85"."""

        try:
            return self._call_api(prompt, valid_labels={"algebraic", "direct", "backtracking"})
        except RuntimeError:
            logger.warning("Reasoning style classification failed, using fallback 'direct'")
            return self._STYLE_FALLBACK, 0.5

    def classify_error_type(
        self,
        question: str,
        response: str,
    ) -> tuple[str, float]:
        """Classify error type using GLM API."""
        prompt = f"""Classify the type of error in this response:

Question: {question}

Response: {response}

Error types:
- arithmetic: correct reasoning approach but numerical calculation mistake
- logic: flawed reasoning chain, wrong formula, or logical fallacy
- hallucination: fabricated numbers, wrong theorem, or unsupported claims
- verification: attempted self-check but failed to catch the error

Respond with just the error type and confidence (0-1), e.g., "arithmetic 0.8"."""

        try:
            return self._call_api(prompt, valid_labels={"arithmetic", "logic", "hallucination", "verification"})
        except RuntimeError:
            logger.warning("Error type classification failed, using fallback 'logic'")
            return self._ERROR_FALLBACK, 0.5

    def _call_api(self, prompt: str, valid_labels: set[str] | None = None) -> tuple[str, float]:
        """Call GLM API with retry logic.

        Args:
            prompt: The prompt to send.
            valid_labels: If provided, the returned label must belong to this set.
                          Responses with unrecognized labels are rejected and retried.

        Raises:
            RuntimeError: If all retry attempts are exhausted or the response
                          cannot be parsed into a valid label.
        """
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
                        "temperature": 0.1,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()["choices"][0]["message"]["content"].strip().lower()

                # Robust parsing: extract label and confidence separately
                label = self._extract_label(result)
                confidence = self._extract_confidence(result)

                if label and (valid_labels is None or label in valid_labels):
                    return label, confidence

                # Label not recognized or not in valid_labels — treat as parse failure
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

    @staticmethod
    def _extract_label(text: str) -> str:
        """Extract a known category label from API response text."""
        known_styles = {"algebraic", "direct", "backtracking"}
        known_errors = {"arithmetic", "logic", "hallucination", "verification"}
        words = text.replace(",", " ").replace(".", " ").split()
        for w in words:
            w_lower = w.lower().strip()
            if w_lower in known_styles or w_lower in known_errors:
                return w_lower
        return ""

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract a confidence score (0-1 float) from API response text."""
        import re
        # Look for patterns like 0.85, .9, 0.9
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

    args = resolve_config(
        cli_args.config, "step02_classify", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )

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

        # Get the final round responses for classification
        debate_rounds = traj.get("debate_rounds", [])
        if debate_rounds:
            final_responses = debate_rounds[-1]
        else:
            final_responses = traj.get("initial_responses", [])

        # Classify reasoning style for correct responses
        # and error type for incorrect responses
        reasoning_styles = []
        error_types = []

        # Determine task type for proper answer comparison
        task_type = traj.get("sample", {}).get("task_type", "math")

        # Per-response classifications (preserve agent-level diversity)
        per_response_labels = []

        for resp in final_responses:
            response_text = resp.get("response", "")
            answer = resp.get("answer")
            agent_name = resp.get("agent_name", "")
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
                "agent_name": agent_name,
                "response": response_text,
                "answer": answer,
                "is_correct": is_correct,
                "reasoning_style": None,
                "reasoning_style_confidence": 0.0,
                "error_type": None,
                "error_type_confidence": 0.0,
            })

            if is_correct:
                # Correct response - classify reasoning style
                style, conf = classifier.classify_reasoning_style(question, response_text)
                reasoning_styles.append((style, conf))
                per_response_labels[-1]["reasoning_style"] = style
                per_response_labels[-1]["reasoning_style_confidence"] = conf
            else:
                # Incorrect response - classify error type
                error, conf = classifier.classify_error_type(question, response_text)
                error_types.append((error, conf))
                per_response_labels[-1]["error_type"] = error
                per_response_labels[-1]["error_type_confidence"] = conf

        # Aggregate classifications for backward compat (sample-level summary)
        from collections import Counter

        reasoning_style = None
        reasoning_confidence = 0.0
        if reasoning_styles:
            style_counter = Counter([s for s, _ in reasoning_styles])
            reasoning_style = style_counter.most_common(1)[0][0]
            reasoning_confidence = sum(c for s, c in reasoning_styles if s == reasoning_style) / len(reasoning_styles)

        error_type = None
        error_confidence = 0.0
        if error_types:
            error_counter = Counter([e for e, _ in error_types])
            error_type = error_counter.most_common(1)[0][0]
            error_confidence = sum(c for e, c in error_types if e == error_type) / len(error_types)

        result = ClassificationResult(
            sample_id=sample_id,
            reasoning_style=reasoning_style,
            reasoning_style_confidence=reasoning_confidence,
            error_type=error_type,
            error_type_confidence=error_confidence,
            metadata={
                "num_correct": len(reasoning_styles),
                "num_incorrect": len(error_types),
            },
        )

        results.append({
            "sample_id": sample_id,
            "reasoning_style": reasoning_style,
            "reasoning_style_confidence": reasoning_confidence,
            "error_type": error_type,
            "error_type_confidence": error_confidence,
            "metadata": result.metadata,
            "per_response_labels": per_response_labels,
        })

        completed_ids.add(sample_id)

    # Save final results
    logger.info("[Step 5] Saving results...")

    output_file = os.path.join(output_dir, "classified_data.json")
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "metadata": {
                "total_trajectories": len(trajectories),
                "api_model": args.api_model,
            },
        }, f, indent=2, ensure_ascii=False)

    # Save per-style splits
    logger.info("[Step 6] Creating per-style splits...")

    style_splits = {}
    error_splits = {}

    # Per-response granular splits: maps (sample_id, response_index) -> style/error
    per_response_style_splits = {}
    per_response_error_splits = {}

    for result in results:
        style = result["reasoning_style"]
        error = result["error_type"]
        sample_id = result["sample_id"]

        if style:
            style_splits.setdefault(style, []).append(sample_id)

        if error:
            error_splits.setdefault(error, []).append(sample_id)

        # Build per-response splits for finer-grained data diversification
        per_labels = result.get("per_response_labels", [])
        for ri, label in enumerate(per_labels):
            if label.get("reasoning_style"):
                per_response_style_splits.setdefault(
                    label["reasoning_style"], [],
                ).append({
                    "sample_id": sample_id,
                    "response_index": ri,
                    "agent_name": label.get("agent_name", ""),
                    "is_correct": label.get("is_correct", False),
                })
            if label.get("error_type"):
                per_response_error_splits.setdefault(
                    label["error_type"], [],
                ).append({
                    "sample_id": sample_id,
                    "response_index": ri,
                    "agent_name": label.get("agent_name", ""),
                    "is_correct": label.get("is_correct", False),
                })

    splits_file = os.path.join(output_dir, "splits.json")
    with open(splits_file, "w") as f:
        json.dump({
            "reasoning_styles": style_splits,
            "error_types": error_splits,
        }, f, indent=2)

    # Save per-response splits for finer-grained Actor/Critic diversification
    per_response_splits_file = os.path.join(output_dir, "per_response_splits.json")
    with open(per_response_splits_file, "w") as f:
        json.dump({
            "reasoning_styles": per_response_style_splits,
            "error_types": per_response_error_splits,
        }, f, indent=2)

    logger.info(f"  Reasoning style splits: {list(style_splits.keys())}")
    logger.info(f"  Error type splits: {list(error_splits.keys())}")
    logger.info(f"  Per-response style splits: { {k: len(v) for k, v in per_response_style_splits.items()} }")
    logger.info(f"  Per-response error splits: { {k: len(v) for k, v in per_response_error_splits.items()} }")

    logger.info("=" * 60)
    logger.info("Classification complete!")
    logger.info(f"  Results: {output_file}")
    logger.info(f"  Splits: {splits_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
