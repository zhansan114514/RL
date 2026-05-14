"""
Diversity split: Critic error-profile routing for Society training.

Incorrect Actor responses are classified by error-profile dimensions and
routed into specialist Critic training mixtures.  Actors are selected for
first-round SFT in scripts/09_train_actors_sft.py, not through this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from src.society.agent_registry import CriticSkill
from src.society.data_classifier import (
    ERROR_PROFILE_DIMENSIONS,
    classify_error_profile,
    ClassificationError,
)

logger = logging.getLogger(__name__)


PROFILE_TO_SKILL = {
    "reasoning": CriticSkill.REASONING,
    "knowledge": CriticSkill.KNOWLEDGE,
    "grounding": CriticSkill.GROUNDING,
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
    Route incorrect responses across error types.

    Uses DataClassifier to label each sample, then groups by label.
    Minority handling belongs in pair construction, not in destructive split
    downsampling.
    """

    def __init__(
        self,
        seed: int = 42,
        use_api: bool = True,
        cache_dir: str = "output/society/classified",
        api_key: str = "",
        api_base: str = "",
        api_model: str = "",
        strict_classification: bool = False,
        max_classification_failure_rate: float = 0.0,
        max_classification_workers: int = 4,
        request_timeout: int | float = 30,
        max_retries: int = 5,
        retry_delay: int | float = 5,
    ):
        self.rng = np.random.default_rng(seed)
        self.use_api = use_api
        self.cache_dir = cache_dir
        self._api_key = api_key
        self._api_base = api_base
        self._api_model = api_model
        self.strict_classification = strict_classification
        self.max_classification_failure_rate = max(0.0, max_classification_failure_rate)
        self.max_classification_workers = max(1, int(max_classification_workers))
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_classification_metrics: dict[str, dict] = {}

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
        attempted_live: int,
        failures: int,
    ) -> None:
        failure_rate = failures / attempted_live if attempted_live else 0.0
        metrics = {
            "total_items": total,
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

    def split_by_error_profile(
        self,
        samples: list[dict],
        responses: list[str],
        correct_answers: list[str] | None = None,
        extracted_answers: list[str] | None = None,
        dataset_name: str = "",
        response_ids: list[str] | None = None,
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
        attempted_live = 0
        failures = 0
        profile_by_index: dict[int, dict] = {}
        live_groups: dict[str, dict] = {}

        for i, (sample, response) in enumerate(zip(samples, responses)):
            question = sample.get("question", "")
            correct = correct_answers[i] if correct_answers and i < len(correct_answers) else sample.get("answer", "")
            extracted = extracted_answers[i] if extracted_answers and i < len(extracted_answers) else ""

            key = _live_profile_key(
                question=question,
                response=response,
                correct=correct,
                extracted=extracted,
                choices=sample.get("choices", ""),
                dataset_name=dataset_name,
                task_type=sample.get("task_type", ""),
                subject=sample.get("subject", sample.get("category", "")),
            )
            live_groups.setdefault(key, {
                "indices": [],
                "sample": sample,
                "response": response,
                "correct": correct,
                "extracted": extracted,
                "dataset_name": dataset_name,
            })["indices"].append(i)

        if live_groups:
            self._require_live_classifier()

        def classify_group(payload: dict) -> tuple[list[int], dict | None, ClassificationError | None]:
            sample = payload["sample"]
            try:
                result = classify_error_profile(
                    response=payload["response"],
                    question=sample.get("question", ""),
                    extracted_answer=payload["extracted"],
                    correct_answer=payload["correct"],
                    choices=sample.get("choices", ""),
                    dataset_name=payload["dataset_name"],
                    task_type=sample.get("task_type", ""),
                    subject=sample.get("subject", sample.get("category", "")),
                    use_api=self.use_api,
                    cache_dir=self.cache_dir,
                    api_key=self._api_key,
                    api_base=self._api_base,
                    api_model=self._api_model,
                    request_timeout=self.request_timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                )
                return payload["indices"], {
                    "format_status": result.format_status,
                    "scores": result.scores,
                    "primary": result.primary,
                    "secondary": result.secondary,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                }, None
            except ClassificationError as e:
                return payload["indices"], None, e

        if live_groups:
            attempted_live = len(live_groups)
            if self.max_classification_workers == 1:
                group_results = [classify_group(payload) for payload in live_groups.values()]
            else:
                group_results = []
                with ThreadPoolExecutor(max_workers=self.max_classification_workers) as executor:
                    future_map = {
                        executor.submit(classify_group, payload): key
                        for key, payload in live_groups.items()
                    }
                    for future in as_completed(future_map):
                        group_results.append(future.result())

            for indices, profile, error in group_results:
                if profile is None:
                    failures += 1
                    if self.strict_classification:
                        logger.error(
                            f"Error profile classification failed for {len(indices)} duplicated item(s): {error}"
                        )
                        continue
                    profile = _ambiguous_profile(str(error))
                    logger.warning(
                        f"Error profile classification failed for {len(indices)} duplicated item(s), "
                        f"routed to general pool: {error}"
                    )
                for idx in indices:
                    profile_by_index[idx] = profile

        for i, (sample, response) in enumerate(zip(samples, responses)):
            profile = profile_by_index.get(i)
            if profile is None:
                continue
            response_id = response_ids[i] if response_ids and i < len(response_ids) else ""

            assignments = assign_error_profile(profile)
            for label, weight in assignments:
                if label in {"general", "format_failure"}:
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
        allow_oversample: bool = False,
        max_repeats_per_item: int = 1,
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
        selected.extend(self._sample_bucket(
            specialty_pool,
            quotas["specialty"],
            "specialty",
            allow_oversample=allow_oversample,
            max_repeats_per_item=max_repeats_per_item,
        ))
        selected.extend(self._sample_bucket(
            general_pool,
            quotas["general"],
            "general",
            allow_oversample=allow_oversample,
            max_repeats_per_item=max_repeats_per_item,
        ))
        selected.extend(self._sample_bucket(
            calibration_pool,
            quotas["calibration"],
            "calibration",
            allow_oversample=allow_oversample,
            max_repeats_per_item=max_repeats_per_item,
        ))
        selected = _dedupe_routed_items(selected)

        if len(selected) < max_items:
            filler = [
                item for item in all_items
                if _routed_item_key(item) not in {
                    _routed_item_key(existing) for existing in selected
                }
            ]
            selected.extend(self._sample_bucket(
                filler,
                max_items - len(selected),
                "general",
                allow_oversample=allow_oversample,
                max_repeats_per_item=max_repeats_per_item,
            ))
            selected = _dedupe_routed_items(selected)

        self.rng.shuffle(selected)
        return selected[:max_items]

    def _sample_bucket(
        self,
        pool: list[RoutedTrainingItem],
        n: int,
        source_bucket: str,
        allow_oversample: bool = False,
        max_repeats_per_item: int = 1,
    ) -> list[RoutedTrainingItem]:
        """Sample a pool and annotate copies with their source bucket."""
        sample_with_replacement = allow_oversample and len(pool) < n
        if sample_with_replacement:
            n = min(n, max(1, int(max_repeats_per_item)) * len(pool))
        sampled = self._sample_pool(pool, n, replace=sample_with_replacement)
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


def assign_error_profile(
    profile: dict,
    min_score: float = 0.35,
    top_k: int = 2,
    min_confidence: float = 0.5,
) -> list[tuple[str, float]]:
    """Assign a profile to one or more error dimensions with weights."""
    primary = str(profile.get("primary", "")).strip().lower() if isinstance(profile, dict) else ""
    if primary == "format_failure":
        return [("format_failure", 1.0)]
    if primary == "ambiguous":
        return [("general", 1.0)]

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

    if confidence < min_confidence:
        return [("general", 1.0)]

    if primary in ERROR_PROFILE_DIMENSIONS:
        return [(primary, 1.0)]

    ranked = sorted(clean_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
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


def _dedupe_routed_items(items: list[RoutedTrainingItem]) -> list[RoutedTrainingItem]:
    deduped: list[RoutedTrainingItem] = []
    seen: set[tuple[str, str | None]] = set()
    for item in items:
        key = _routed_item_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _routed_item_key(item: RoutedTrainingItem) -> tuple[str, str | None]:
    base = item.response_id or _content_hash(
        item.sample.get("question", ""),
        item.response,
    )
    skill = item.skill.value if item.skill else None
    return (base, skill)


def _ambiguous_profile(evidence: str = "") -> dict:
    return {
        "format_status": "empty_or_invalid",
        "scores": {dim: 0.0 for dim in ERROR_PROFILE_DIMENSIONS},
        "primary": "ambiguous",
        "secondary": [],
        "confidence": 0.0,
        "evidence": evidence,
    }


def _content_hash(question: str, response: str) -> str:
    """Stable hash for (question, response) pair — used for lookup keys."""
    content = f"{question}||{response}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _live_profile_key(
    question: str,
    response: str,
    correct: str,
    extracted: str,
    choices: str | list | dict = "",
    dataset_name: str = "",
    task_type: str = "",
    subject: str = "",
) -> str:
    """Stable key for deduplicating live error-profile classifications."""
    payload = {
        "question": question,
        "response": response,
        "correct": correct,
        "extracted": extracted,
        "choices": choices,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "subject": subject,
    }
    content = json.dumps(payload, sort_keys=True, ensure_ascii=False)
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
