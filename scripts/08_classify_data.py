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

from _utils import load_yaml_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "input_dir", "output_dir")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "cache/society",
    "input_dir": "cache/society/bootstrap",
    "output_dir": "cache/society/classified",
    "api_key": "bcf988da32f64948a82fd7dda3b9b3d3.mVYoCk3Wi5ZrcsUM",
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

    def classify_reasoning_style(
        self,
        question: str,
        response: str,
    ) -> tuple[str, float]:
        """Classify reasoning style using GLM API."""
        prompt = f"""Classify the reasoning style in this response into one of:
- analytical: step-by-step logical decomposition
- intuitive: pattern recognition, quick judgment
- creative: lateral thinking, novel approaches
- skeptical: verification, fact-checking

Question: {question}

Response: {response}

Respond with just the style name and a confidence score (0-1), e.g., "analytical 0.85"."""

        return self._call_api(prompt)

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
- arithmetic: calculation mistakes
- logic: logical fallacies or invalid deductions
- hallucination: factually incorrect claims
- verification: needs verification
- interpretation: misunderstanding the question
- completeness: incomplete solution

Respond with just the error type and confidence (0-1), e.g., "arithmetic 0.8"."""

        return self._call_api(prompt)

    def _call_api(self, prompt: str) -> tuple[str, float]:
        """Call GLM API with retry logic."""
        import requests

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
                parts = result.split()

                if len(parts) >= 2:
                    return parts[0], float(parts[1])
                else:
                    return result, 0.5

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts, using fallback")
                    return "analytical", 0.5


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

    # Load config
    cfg = load_yaml_config(cli_args.config)
    common_cfg = cfg.get("common", {})
    step_cfg = cfg.get("step02_classify", {})

    merged = dict(STEP_DEFAULTS)
    for key in COMMON_KEYS:
        if key in common_cfg and common_cfg[key] is not None:
            merged[key] = common_cfg[key]
    for key in STEP_DEFAULTS:
        if key in step_cfg and step_cfg[key] is not None:
            merged[key] = step_cfg[key]

    # CLI override for API key
    if cli_args.api_key:
        merged["api_key"] = cli_args.api_key

    merged["config"] = cli_args.config

    return argparse.Namespace(**merged)


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

        for resp in final_responses:
            response_text = resp.get("response", "")
            answer = resp.get("answer")
            correct_answer = traj.get("consensus_answer", "")

            if answer and answer == correct_answer:
                # Correct response - classify reasoning style
                style, conf = classifier.classify_reasoning_style(question, response_text)
                reasoning_styles.append((style, conf))
            else:
                # Incorrect response - classify error type
                error, conf = classifier.classify_error_type(question, response_text)
                error_types.append((error, conf))

        # Aggregate classifications (use most common)
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

    for result in results:
        style = result["reasoning_style"]
        error = result["error_type"]

        if style:
            style_splits.setdefault(style, []).append(result["sample_id"])

        if error:
            error_splits.setdefault(error, []).append(result["sample_id"])

    splits_file = os.path.join(output_dir, "splits.json")
    with open(splits_file, "w") as f:
        json.dump({
            "reasoning_styles": style_splits,
            "error_types": error_splits,
        }, f, indent=2)

    logger.info(f"  Reasoning style splits: {list(style_splits.keys())}")
    logger.info(f"  Error type splits: {list(error_splits.keys())}")

    logger.info("=" * 60)
    logger.info("Classification complete!")
    logger.info(f"  Results: {output_file}")
    logger.info(f"  Splits: {splits_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
