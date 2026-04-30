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
import os
from collections import Counter
from dataclasses import dataclass, replace
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
    source_bucket: str = "unknown"
    response_id: str = ""


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
        api_key: str = "",
        api_base: str = "",
        api_model: str = "",
        strict_classification: bool = False,
        max_classification_failure_rate: float = 0.0,
    ):
        self.balance = balance
        self.rng = np.random.default_rng(seed)
        self.use_api = use_api
        self.cache_dir = cache_dir
        self.pre_classified_file = pre_classified_file
        self._pre_classified: Optional[dict] = None
        self._api_key = api_key
        self._api_base = api_base
        self._api_model = api_model
        self.strict_classification = strict_classification
        self.max_classification_failure_rate = max(0.0, max_classification_failure_rate)
        self.last_classification_metrics: dict[str, dict] = {}

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

        # Build lookups by stable response_id first, with content hash as a
        # secondary key for generated pairs that do not originate from bootstrap.
        by_response_id: dict[str, dict] = {}
        by_content: dict[str, dict] = {}
        for result in data.get("results", []):
            # Use per-response labels for fine-grained matching
            for label in result.get("per_response_labels", []):
                question = result.get("question") or result.get("sample_id", "")
                response = label.get("response", "")
                key = _content_hash(question, response)
                entry = {}
                if label.get("response_id"):
                    entry["response_id"] = label["response_id"]
                if label.get("reasoning_style"):
                    entry["style"] = label["reasoning_style"]
                    entry["style_confidence"] = label.get("reasoning_style_confidence", 0.5)
                if label.get("error_profile"):
                    entry["error_profile"] = label["error_profile"]
                if entry:
                    by_content[key] = entry
                    if entry.get("response_id"):
                        by_response_id[entry["response_id"]] = entry

        logger.info(
            f"Loaded {len(by_content)} pre-classified entries from {path}"
        )
        return {"by_response_id": by_response_id, "by_content": by_content}

    def _lookup_pre_classified_style(
        self, question: str, response: str, response_id: str = "",
    ) -> Optional[ReasoningStyle]:
        """Check pre-classified results for a reasoning style match."""
        if not self._pre_classified:
            return None
        entry = None
        if response_id:
            entry = self._pre_classified.get("by_response_id", {}).get(response_id)
        key = _content_hash(question, response)
        if entry is None:
            entry = self._pre_classified.get("by_content", {}).get(key)
        if entry and "style" in entry:
            try:
                return ReasoningStyle(entry["style"])
            except ValueError:
                pass
        return None

    def _lookup_pre_classified_error_profile(
        self, question: str, response: str, response_id: str = "",
    ) -> Optional[dict]:
        """Check pre-classified results for an error profile match."""
        if not self._pre_classified:
            return None
        entry = None
        if response_id:
            entry = self._pre_classified.get("by_response_id", {}).get(response_id)
        key = _content_hash(question, response)
        if entry is None:
            entry = self._pre_classified.get("by_content", {}).get(key)
        if entry and "error_profile" in entry:
            return entry["error_profile"]
        return None

    def _require_live_classifier(self) -> None:
        if not self.strict_classification:
            return
        if not self.use_api:
            raise ClassificationError(
                "strict_classification=True requires use_api=True for "
                "unseen responses"
            )
        if not (self._api_key or os.environ.get("GLM_API_KEY")):
            raise ClassificationError(
                "strict_classification=True requires GLM_API_KEY or api_key "
                "for unseen responses"
            )

    def _record_classification_metrics(
        self,
        kind: str,
        total: int,
        preclassified: int,
        attempted_live: int,
        failures: int,
    ) -> None:
        failure_rate = failures / attempted_live if attempted_live else 0.0
        metrics = {
            "total_items": total,
            "preclassified": preclassified,
            "attempted_live": attempted_live,
            "failures": failures,
            "failure_rate": failure_rate,
            "strict_classification": self.strict_classification,
            "max_classification_failure_rate": self.max_classification_failure_rate,
        }
        self.last_classification_metrics[kind] = metrics
        if (
            self.strict_classification
            and attempted_live
            and failure_rate > self.max_classification_failure_rate
        ):
            raise ClassificationError(
                f"{kind} classification failure rate {failure_rate:.3f} "
                f"exceeds threshold {self.max_classification_failure_rate:.3f} "
                f"({failures}/{attempted_live})"
            )

    def split_by_reasoning_style(
        self,
        samples: list[dict],
        responses: Optional[list[str]] = None,
        answers: Optional[list[str]] = None,
        response_ids: Optional[list[str]] = None,
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
        preclassified = 0
        attempted_live = 0
        failures = 0

        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            response = responses[i] if responses and i < len(responses) else ""
            answer = answers[i] if answers and i < len(answers) else sample.get("answer", "")
            response_id = response_ids[i] if response_ids and i < len(response_ids) else ""

            if response:
                # Try pre-classified data first (single source of truth)
                style = self._lookup_pre_classified_style(question, response, response_id)
                if style is not None:
                    preclassified += 1
                    splits[style].append(sample)
                    continue

                # Fall back to live classification
                self._require_live_classifier()
                try:
                    attempted_live += 1
                    result = classify_reasoning_style(
                        response=response,
                        question=question,
                        correct_answer=answer,
                        use_api=self.use_api,
                        cache_dir=self.cache_dir,
                        api_key=self._api_key,
                        api_base=self._api_base,
                        api_model=self._api_model,
                    )
                    style = result.style
                except ClassificationError as e:
                    failures += 1
                    if self.strict_classification:
                        logger.error(f"Reasoning style classification failed for sample {i}: {e}")
                        continue
                    style = list(ReasoningStyle)[i % len(ReasoningStyle)]
                    logger.warning(
                        f"Classification failed for sample {i}, "
                        f"assigned style '{style.value}' via round-robin fallback: {e}"
                    )
            else:
                if self.strict_classification:
                    attempted_live += 1
                    failures += 1
                    logger.error(f"Missing response for strict reasoning style classification at sample {i}")
                    continue
                style = list(ReasoningStyle)[i % len(ReasoningStyle)]

            splits[style].append(sample)

        self._record_classification_metrics(
            "reasoning_style",
            total=len(samples),
            preclassified=preclassified,
            attempted_live=attempted_live,
            failures=failures,
        )

        if self.balance:
            splits = self._balance_splits(splits)

        logger.info(
            "Reasoning style split: "
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
        response_ids: Optional[list[str]] = None,
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
        preclassified = 0
        attempted_live = 0
        failures = 0

        for i, (sample, response) in enumerate(zip(samples, responses)):
            question = sample.get("question", "")
            correct = correct_answers[i] if correct_answers and i < len(correct_answers) else sample.get("answer", "")
            extracted = extracted_answers[i] if extracted_answers and i < len(extracted_answers) else ""
            response_id = response_ids[i] if response_ids and i < len(response_ids) else ""

            # Try pre-classified data first (single source of truth)
            profile = self._lookup_pre_classified_error_profile(question, response, response_id)
            if profile is not None:
                preclassified += 1

            # Fall back to live classification
            if profile is None:
                self._require_live_classifier()
                try:
                    attempted_live += 1
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
                        api_key=self._api_key,
                        api_base=self._api_base,
                        api_model=self._api_model,
                    )
                    profile = {
                        "scores": result.scores,
                        "primary": result.primary,
                        "secondary": result.secondary,
                        "confidence": result.confidence,
                        "evidence": result.evidence,
                    }
                except ClassificationError as e:
                    failures += 1
                    if self.strict_classification:
                        logger.error(f"Error profile classification failed for sample {i}: {e}")
                        continue
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
                    response_id=response_id,
                ))

        self._record_classification_metrics(
            "error_profile",
            total=len(responses),
            preclassified=preclassified,
            attempted_live=attempted_live,
            failures=failures,
        )

        logger.info(
            "Error profile routing: "
            + ", ".join(f"{k}={v}" for k, v in profile_counts.items())
        )
        return routed_items

    def build_critic_training_mix(
        self,
        all_items: list[RoutedTrainingItem],
        target_skill: CriticSkill,
        max_items: int = 512,
        min_specialty_items: int = 32,
        min_specialty_ratio: float = 0.08,
        specialty_ratio: float = 0.7,
        general_ratio: float = 0.2,
        calibration_ratio: float = 0.1,
    ) -> list[RoutedTrainingItem]:
        """
        Build a specialist-heavy training mix only when enough target data exists.

        The specialty pool is the gate for specialist training. If the target
        skill is too sparse, callers should skip or freeze that critic rather
        than train it on mostly unrelated general data.
        """
        if not all_items or max_items <= 0:
            return []

        specialty_pool = [item for item in all_items if item.skill == target_skill]
        specialty_count = len(specialty_pool)
        specialty_ratio_actual = specialty_count / max(len(all_items), 1)
        if (
            specialty_count < min_specialty_items
            or specialty_ratio_actual < min_specialty_ratio
        ):
            logger.info(
                "Skipping specialist mix for %s: specialty=%s/%s (%.3f), "
                "thresholds min_items=%s min_ratio=%.3f",
                target_skill.value,
                specialty_count,
                len(all_items),
                specialty_ratio_actual,
                min_specialty_items,
                min_specialty_ratio,
            )
            return []

        general_pool = list(all_items)
        calibration_pool = [
            item for item in all_items
            if item.skill is not None and item.skill != target_skill
        ]

        quota_total = general_ratio + specialty_ratio + calibration_ratio
        if quota_total <= 0:
            quota_total = 1.0

        quotas = {
            "specialty": int(round(max_items * specialty_ratio / quota_total)),
            "general": int(round(max_items * general_ratio / quota_total)),
            "calibration": int(round(max_items * calibration_ratio / quota_total)),
        }
        quota_sum = sum(quotas.values())
        if quota_sum != max_items:
            quotas["specialty"] += max_items - quota_sum

        selected: list[RoutedTrainingItem] = []
        selected.extend(self._sample_bucket(specialty_pool, quotas["specialty"], "specialty"))
        selected.extend(self._sample_bucket(general_pool, quotas["general"], "general"))
        selected.extend(self._sample_bucket(calibration_pool, quotas["calibration"], "calibration"))

        if len(selected) < max_items:
            filler = specialty_pool or general_pool
            selected.extend(self._sample_bucket(
                filler,
                max_items - len(selected),
                "specialty" if filler is specialty_pool else "general",
            ))

        self.rng.shuffle(selected)
        return selected[:max_items]

    def _sample_bucket(
        self,
        pool: list[RoutedTrainingItem],
        n: int,
        source_bucket: str,
    ) -> list[RoutedTrainingItem]:
        """Sample a pool and annotate copies with their source bucket."""
        sampled = self._sample_pool(pool, n, replace=len(pool) < n)
        return [replace(item, source_bucket=source_bucket) for item in sampled]

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


def summarize_critic_training_pairs(preference_pairs: list[dict]) -> dict:
    """Summarize selected critic pairs for experiment reporting."""
    total = len(preference_pairs)
    unique_pairs = {
        (
            p.get("sample", {}).get("question", ""),
            p.get("actor_response", ""),
        )
        for p in preference_pairs
    }
    bucket_counts = Counter(
        p.get("metadata", {}).get("source_bucket", "unknown")
        for p in preference_pairs
    )
    assigned_skill_counts = Counter(
        p.get("metadata", {}).get("assigned_skill", "unknown")
        for p in preference_pairs
    )
    bucket_ratios = {
        bucket: (bucket_counts.get(bucket, 0) / total if total else 0.0)
        for bucket in ("general", "specialty", "calibration")
    }
    return {
        "sample_count": total,
        "unique_pair_count": len(unique_pairs),
        "duplicate_rate": 1.0 - (len(unique_pairs) / total) if total else 0.0,
        "source_bucket_counts": dict(bucket_counts),
        "source_bucket_ratios": bucket_ratios,
        "assigned_skill_counts": dict(assigned_skill_counts),
    }
