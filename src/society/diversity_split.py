"""
Diversity split: Data-level diversification for Multiagent FT.

Partitions training data by:
1. Reasoning style (ALGEBRAIC, DIRECT, BACKTRACKING) → for Actor diversification
2. Error profile dimensions → weighted Critic-skill routing

Each Actor trains only on its style subset.  Critics receive a mixture of
general examples, high-relevance specialty examples, and out-of-specialty
calibration examples.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.society.agent_registry import ReasoningStyle, CriticSkill
from src.society.data_classifier import (
    ERROR_PROFILE_DIMENSIONS,
    classify_reasoning_style,
    classify_error_profile,
    ClassificationError,
)

logger = logging.getLogger(__name__)


PROFILE_TO_SKILL = {
    "computation": CriticSkill.COMPUTATION,
    "reasoning": CriticSkill.REASONING,
    "knowledge": CriticSkill.KNOWLEDGE,
    "grounding": CriticSkill.KNOWLEDGE,
    "verification": CriticSkill.VERIFICATION,
}


@dataclass
class RoutedTrainingItem:
    """Weighted assignment of one incorrect response to a critic skill."""
    sample: dict
    response: str
    skill: CriticSkill | None
    weight: float
    profile: dict


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

        # Build lookup: (question_hash, response_hash) -> classification payload.
        lookup: dict[tuple[str, str], dict] = {}
        for result in data.get("results", []):
            # Use per-response labels for fine-grained matching
            for label in result.get("per_response_labels", []):
                question = result.get("question") or result.get("sample_id", "")
                response = label.get("response", "")
                key = _content_hash(question, response)
                entry = {}
                if label.get("reasoning_style"):
                    entry["style"] = label["reasoning_style"]
                    entry["style_confidence"] = label.get("reasoning_style_confidence", 0.5)
                if label.get("error_profile"):
                    entry["error_profile"] = label["error_profile"]
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

    def _lookup_pre_classified_error_profile(
        self, question: str, response: str,
    ) -> Optional[dict]:
        """Check pre-classified results for an error profile match."""
        if not self._pre_classified:
            return None
        key = _content_hash(question, response)
        entry = self._pre_classified.get(key)
        if entry and "error_profile" in entry:
            return entry["error_profile"]
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

    def split_by_error_profile(
        self,
        samples: list[dict],
        responses: list[str],
        correct_answers: Optional[list[str]] = None,
        extracted_answers: Optional[list[str]] = None,
        dataset_name: str = "",
    ) -> list[RoutedTrainingItem]:
        """
        Route incorrect responses by multi-dimensional error profile.

        A response may be assigned to up to two critic skills with normalized
        weights.  Low-confidence profiles are routed to the general pool
        (`skill=None`) instead of being assigned to a default reasoning bucket.

        Args:
            samples: List of standardized sample dicts.
            responses: Corresponding (incorrect) responses.
            correct_answers: Optional correct answers.
            extracted_answers: Optional answers extracted from responses.
            dataset_name: Dataset label for the classifier prompt.

        Returns:
            Weighted routed training items.
        """
        routed_items: list[RoutedTrainingItem] = []
        profile_counts = {dim: 0 for dim in ERROR_PROFILE_DIMENSIONS}
        profile_counts["general"] = 0

        for i, (sample, response) in enumerate(zip(samples, responses)):
            question = sample.get("question", "")
            correct = correct_answers[i] if correct_answers and i < len(correct_answers) else sample.get("answer", "")
            extracted = extracted_answers[i] if extracted_answers and i < len(extracted_answers) else ""

            # Try pre-classified data first (single source of truth)
            profile = self._lookup_pre_classified_error_profile(question, response)

            # Fall back to live classification
            if profile is None:
                try:
                    result = classify_error_profile(
                        response=response,
                        question=question,
                        extracted_answer=extracted,
                        correct_answer=correct,
                        choices=sample.get("choices", ""),
                        dataset_name=dataset_name,
                        task_type=sample.get("task_type", ""),
                        subject=sample.get("subject", sample.get("category", "")),
                        use_api=self.use_api,
                        cache_dir=self.cache_dir,
                    )
                    profile = {
                        "scores": result.scores,
                        "primary": result.primary,
                        "secondary": result.secondary,
                        "confidence": result.confidence,
                        "evidence": result.evidence,
                    }
                except ClassificationError as e:
                    profile = _unknown_profile(str(e))
                    logger.warning(
                        f"Error profile classification failed for sample {i}, "
                        f"routed to general pool: {e}"
                    )

            assignments = assign_error_profile(profile)
            for label, weight in assignments:
                if label == "general":
                    profile_counts["general"] += 1
                    skill = None
                else:
                    profile_counts[label] += 1
                    skill = PROFILE_TO_SKILL[label]
                routed_items.append(RoutedTrainingItem(
                    sample=sample,
                    response=response,
                    skill=skill,
                    weight=weight,
                    profile=profile,
                ))

        logger.info(
            f"Error profile routing: "
            + ", ".join(f"{k}={v}" for k, v in profile_counts.items())
        )
        return routed_items

    def build_critic_training_mix(
        self,
        all_items: list[RoutedTrainingItem],
        target_skill: CriticSkill,
        general_ratio: float = 0.6,
        specialty_ratio: float = 0.3,
        calibration_ratio: float = 0.1,
        max_items: int = 512,
    ) -> list[RoutedTrainingItem]:
        """Build shared-base + specialty + calibration training items."""
        if not all_items or max_items <= 0:
            return []

        specialty_pool = [item for item in all_items if item.skill == target_skill]
        general_pool = list(all_items)
        calibration_pool = [
            item for item in all_items
            if item.skill is not None and item.skill != target_skill
        ]

        quota_total = general_ratio + specialty_ratio + calibration_ratio
        if quota_total <= 0:
            quota_total = 1.0

        quotas = {
            "general": int(round(max_items * general_ratio / quota_total)),
            "specialty": int(round(max_items * specialty_ratio / quota_total)),
            "calibration": int(round(max_items * calibration_ratio / quota_total)),
        }
        quota_sum = sum(quotas.values())
        if quota_sum != max_items:
            quotas["general"] += max_items - quota_sum

        selected = []
        selected.extend(self._sample_pool(general_pool, quotas["general"], replace=len(general_pool) < quotas["general"]))
        selected.extend(self._sample_pool(specialty_pool, quotas["specialty"], replace=len(specialty_pool) < quotas["specialty"]))
        selected.extend(self._sample_pool(calibration_pool, quotas["calibration"], replace=len(calibration_pool) < quotas["calibration"]))

        if len(selected) < max_items:
            filler = specialty_pool or general_pool
            selected.extend(self._sample_pool(
                filler,
                max_items - len(selected),
                replace=len(filler) < (max_items - len(selected)),
            ))

        self.rng.shuffle(selected)
        return selected[:max_items]

    def _sample_pool(
        self,
        pool: list[RoutedTrainingItem],
        n: int,
        replace: bool = False,
    ) -> list[RoutedTrainingItem]:
        """Sample a pool, optionally oversampling small minority pools."""
        if n <= 0 or not pool:
            return []
        size = n if replace else min(n, len(pool))
        indices = self.rng.choice(len(pool), size=size, replace=replace)
        return [pool[int(i)] for i in indices]

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


def assign_error_profile(
    profile: dict,
    min_score: float = 0.35,
    top_k: int = 2,
) -> list[tuple[str, float]]:
    """Assign a profile to one or more error dimensions with weights."""
    scores = profile.get("scores", {}) if isinstance(profile, dict) else {}
    clean_scores = {}
    for dim in ERROR_PROFILE_DIMENSIONS:
        try:
            clean_scores[dim] = max(0.0, min(1.0, float(scores.get(dim, 0.0))))
        except (TypeError, ValueError):
            clean_scores[dim] = 0.0

    confidence = profile.get("confidence", 0.0) if isinstance(profile, dict) else 0.0
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    ranked = sorted(clean_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or confidence < 0.5:
        return [("general", 1.0)]

    top1, s1 = ranked[0]
    top2, s2 = ranked[1] if len(ranked) > 1 else ("general", 0.0)

    if s1 >= 0.6 and s1 - s2 >= 0.15:
        return [(top1, 1.0)]

    selected = [(label, score) for label, score in ranked[:top_k] if score >= min_score]
    total = sum(score for _, score in selected)
    if total <= 0:
        return [("general", 1.0)]

    return [(label, score / total) for label, score in selected]


def _unknown_profile(evidence: str = "") -> dict:
    return {
        "scores": {dim: 0.0 for dim in ERROR_PROFILE_DIMENSIONS},
        "primary": "unknown",
        "secondary": [],
        "confidence": 0.0,
        "evidence": evidence,
    }


def _content_hash(question: str, response: str) -> str:
    """Stable hash for (question, response) pair — used for lookup keys."""
    content = f"{question}||{response}"
    return hashlib.md5(content.encode()).hexdigest()[:12]
