"""
Diversity split: Data-level diversification for Multiagent FT.

Partitions training data by:
1. Reasoning style (ALGEBRAIC, DIRECT, BACKTRACKING) → for Actor diversification
2. Error type (ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION) → for Critic diversification

Each Actor trains only on its style subset; each Critic trains only on its error-type subset.
This data-level diversification naturally produces diverse thinking chains.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.society.agent_registry import ReasoningStyle, ErrorType
from src.society.data_classifier import classify_reasoning_style, classify_error_type, ClassificationError

logger = logging.getLogger(__name__)


class DiversitySplit:
    """
    Split training data across reasoning styles and error types.

    Uses DataClassifier to label each sample, then groups by label.
    Optionally balances splits by downsampling majority groups.

    Supports loading pre-classified results from phase 2 (08_classify_data.py)
    via `pre_classified_file` to avoid redundant re-classification.
    """

    def __init__(
        self,
        balance: bool = True,
        seed: int = 42,
        use_api: bool = True,
        cache_dir: str = "output/society/classified",
        pre_classified_file: Optional[str] = None,
    ):
        self.balance = balance
        self.rng = np.random.default_rng(seed)
        self.use_api = use_api
        self.cache_dir = cache_dir
        self.pre_classified_file = pre_classified_file
        self._pre_classified: Optional[dict] = None

        if pre_classified_file:
            self._pre_classified = self._load_pre_classified(pre_classified_file)

    @staticmethod
    def _load_pre_classified(path: str) -> Optional[dict]:
        """Load pre-classified results from phase 2 (08_classify_data.py).

        Returns a dict keyed by (question, response) hash -> classification,
        or None if the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Pre-classified file not found: {path}")
            return None

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load pre-classified file: {e}")
            return None

        # Build lookup: (question_hash, response_hash) -> {style, error_type, confidence}
        lookup: dict[tuple[str, str], dict] = {}
        for result in data.get("results", []):
            # Use per-response labels for fine-grained matching
            for label in result.get("per_response_labels", []):
                question = result.get("sample_id", "")
                response = label.get("response", "")
                key = _content_hash(question, response)
                entry = {}
                if label.get("reasoning_style"):
                    entry["style"] = label["reasoning_style"]
                    entry["style_confidence"] = label.get("reasoning_style_confidence", 0.5)
                if label.get("error_type"):
                    entry["error_type"] = label["error_type"]
                    entry["error_confidence"] = label.get("error_type_confidence", 0.5)
                if entry:
                    lookup[key] = entry

        logger.info(
            f"Loaded {len(lookup)} pre-classified entries from {path}"
        )
        return lookup

    def _lookup_pre_classified_style(
        self, question: str, response: str,
    ) -> Optional[ReasoningStyle]:
        """Check pre-classified results for a reasoning style match."""
        if not self._pre_classified:
            return None
        key = _content_hash(question, response)
        entry = self._pre_classified.get(key)
        if entry and "style" in entry:
            try:
                return ReasoningStyle(entry["style"])
            except ValueError:
                pass
        return None

    def _lookup_pre_classified_error(
        self, question: str, response: str,
    ) -> Optional[ErrorType]:
        """Check pre-classified results for an error type match."""
        if not self._pre_classified:
            return None
        key = _content_hash(question, response)
        entry = self._pre_classified.get(key)
        if entry and "error_type" in entry:
            try:
                return ErrorType(entry["error_type"])
            except ValueError:
                pass
        return None

    def split_by_reasoning_style(
        self,
        samples: list[dict],
        responses: Optional[list[str]] = None,
        answers: Optional[list[str]] = None,
    ) -> dict[ReasoningStyle, list[dict]]:
        """
        Split samples by reasoning style.

        For samples with correct responses, classifies the reasoning style
        and groups accordingly.

        Args:
            samples: List of standardized sample dicts.
            responses: Optional corresponding responses (for classification).
            answers: Optional correct answers.

        Returns:
            Dict mapping ReasoningStyle to list of samples.
        """
        splits: dict[ReasoningStyle, list[dict]] = {s: [] for s in ReasoningStyle}

        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            response = responses[i] if responses and i < len(responses) else ""
            answer = answers[i] if answers and i < len(answers) else sample.get("answer", "")

            if response:
                # Try pre-classified data first (single source of truth)
                style = self._lookup_pre_classified_style(question, response)
                if style is not None:
                    splits[style].append(sample)
                    continue

                # Fall back to live classification
                try:
                    result = classify_reasoning_style(
                        response=response,
                        question=question,
                        correct_answer=answer,
                        use_api=self.use_api,
                        cache_dir=self.cache_dir,
                    )
                    style = result.style
                except ClassificationError as e:
                    style = list(ReasoningStyle)[i % len(ReasoningStyle)]
                    logger.warning(
                        f"Classification failed for sample {i}, "
                        f"assigned style '{style.value}' via round-robin fallback: {e}"
                    )
            else:
                # Round-robin assignment if no responses available
                style = list(ReasoningStyle)[i % len(ReasoningStyle)]

            splits[style].append(sample)

        if self.balance:
            splits = self._balance_splits(splits)

        logger.info(
            f"Reasoning style split: "
            + ", ".join(f"{s.value}={len(v)}" for s, v in splits.items())
        )
        return splits

    def split_by_error_type(
        self,
        samples: list[dict],
        responses: list[str],
        correct_answers: Optional[list[str]] = None,
        extracted_answers: Optional[list[str]] = None,
    ) -> dict[ErrorType, list[tuple[dict, str]]]:
        """
        Split samples by error type (for Critic training data).

        For samples with incorrect responses, classifies the error type
        and groups accordingly.

        Args:
            samples: List of standardized sample dicts.
            responses: Corresponding (incorrect) responses.
            correct_answers: Optional correct answers.
            extracted_answers: Optional answers extracted from responses.

        Returns:
            Dict mapping ErrorType to list of (sample, response) tuples.
        """
        splits: dict[ErrorType, list[tuple[dict, str]]] = {e: [] for e in ErrorType}

        for i, (sample, response) in enumerate(zip(samples, responses)):
            question = sample.get("question", "")
            correct = correct_answers[i] if correct_answers and i < len(correct_answers) else sample.get("answer", "")
            extracted = extracted_answers[i] if extracted_answers and i < len(extracted_answers) else ""

            # Try pre-classified data first (single source of truth)
            error_type = self._lookup_pre_classified_error(question, response)
            if error_type is not None:
                splits[error_type].append((sample, response))
                continue

            # Fall back to live classification
            try:
                result = classify_error_type(
                    response=response,
                    question=question,
                    extracted_answer=extracted,
                    correct_answer=correct,
                    use_api=self.use_api,
                    cache_dir=self.cache_dir,
                )
                splits[result.error_type].append((sample, response))
            except ClassificationError as e:
                error_type = list(ErrorType)[i % len(ErrorType)]
                logger.warning(
                    f"Error classification failed for sample {i}, "
                    f"assigned error type '{error_type.value}' via round-robin fallback: {e}"
                )
                splits[error_type].append((sample, response))

        if self.balance:
            splits = self._balance_splits(splits)

        logger.info(
            f"Error type split: "
            + ", ".join(f"{e.value}={len(v)}" for e, v in splits.items())
        )
        return splits

    def _balance_splits(self, splits: dict) -> dict:
        """Downsample majority groups to match minority."""
        non_empty = {k: v for k, v in splits.items() if v}
        if not non_empty:
            return splits

        target = min(len(v) for v in non_empty.values())
        balanced = {}
        for key, items in splits.items():
            if len(items) > target:
                indices = self.rng.choice(len(items), size=target, replace=False)
                balanced[key] = [items[i] for i in indices]
            else:
                balanced[key] = items
        return balanced


def _content_hash(question: str, response: str) -> str:
    """Stable hash for (question, response) pair — used for lookup keys."""
    content = f"{question}||{response}"
    return hashlib.md5(content.encode()).hexdigest()[:12]
