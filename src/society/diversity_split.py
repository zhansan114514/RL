"""
Diversity split: Data-level diversification for Multiagent FT.

Partitions training data by:
1. Reasoning style (ALGEBRAIC, DIRECT, BACKTRACKING) → for Actor diversification
2. Error type (ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION) → for Critic diversification

Each Actor trains only on its style subset; each Critic trains only on its error-type subset.
This data-level diversification naturally produces diverse thinking chains.
"""

from __future__ import annotations

import logging
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
    """

    def __init__(
        self,
        balance: bool = True,
        seed: int = 42,
        use_api: bool = True,
        cache_dir: str = "output/society/classified",
    ):
        self.balance = balance
        self.rng = np.random.default_rng(seed)
        self.use_api = use_api
        self.cache_dir = cache_dir

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
