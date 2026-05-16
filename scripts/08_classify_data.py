"""
Classify Actor SFT candidates with the shared MMLU-aware style classifier.

Phase 08 now performs local correctness checks for every generated response,
then calls the reasoning-style classifier only for correct responses.  The main
training gate is accepted_for_actor_sft: correct, trainable, prompted-style
matched, classified-style matched, and above the configured confidence floor.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.society.agent_registry import ReasoningStyle
from src.society.actor_response_quality import is_trainable_actor_response
from src.society.data_classifier import (
    ClassificationError,
    STYLE_CLASSIFIER_VERSION,
    classify_reasoning_style,
)
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "api_key": "",
    "api_base": "https://api.labforge.top",
    "api_model": "gpt5.5",
    "batch_size": 10,
    "request_timeout": 30,
    "retry_delay": 5,
    "max_retries": 5,
    "input_dir": "output/society/bootstrap",
    "output_dir": "output/society/classified",
    "strict_classification": True,
    "max_classification_failure_rate": 0.0,
    "max_workers": 4,
    "min_style_confidence": 0.60,
}

VALID_STYLES = {style.value for style in ReasoningStyle}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Classify bootstrap data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/society/experiment_mmlu.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--api_key", type=str, default=None, help="API key override.")
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step02_classify", defaults=STEP_DEFAULTS).to_namespace()
    if cli_args.api_key:
        args.api_key = cli_args.api_key
    elif not args.api_key:
        args.api_key = (
            os.environ.get("GPT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GLM_API_KEY", "")
        )
    return args


def resolve_trajectory_file(input_dir: str) -> Path:
    import glob

    pattern = os.path.join(input_dir, "trajectories.jsonl")
    if not os.path.exists(pattern):
        files = glob.glob(os.path.join(input_dir, "*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No trajectories found in {input_dir}")
        pattern = files[0]
    return Path(pattern)


def fingerprint_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def load_trajectories(input_dir: str) -> list[dict]:
    pattern = resolve_trajectory_file(input_dir)
    trajectories = []
    with open(pattern) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    logger.info("Loaded %s trajectories from %s", len(trajectories), pattern)
    return trajectories


def checkpoint_metadata(args: Any) -> dict[str, Any]:
    trajectory_file = resolve_trajectory_file(str(getattr(args, "input_dir", "")))
    return {
        "schema_version": 4,
        "classification_mode": "actor_sft_selection",
        "input_dir": str(getattr(args, "input_dir", "")),
        "input_file": fingerprint_file(trajectory_file),
        "reasoning_styles": sorted(VALID_STYLES),
        "style_classifier_version": STYLE_CLASSIFIER_VERSION,
        "min_style_confidence": float(getattr(args, "min_style_confidence", 0.65)),
    }


def load_checkpoint(output_dir: str, args: Any) -> dict[str, Any]:
    path = Path(output_dir) / "classification_checkpoint.json"
    if path.exists():
        with open(path) as f:
            checkpoint = json.load(f)
        if checkpoint.get("metadata") == checkpoint_metadata(args):
            return checkpoint
        logger.warning(
            "Ignoring stale classification checkpoint at %s: metadata mismatch",
            path,
        )
    return {"schema_version": 4, "completed": [], "results": []}


def save_checkpoint(output_dir: str, checkpoint: dict[str, Any], args: Any) -> None:
    path = Path(output_dir) / "classification_checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(checkpoint)
    checkpoint["metadata"] = checkpoint_metadata(args)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def make_response_id(sample_id: str, round_num: int, agent_id: int) -> str:
    return f"{sample_id}_round_{round_num}_agent_{agent_id}"


def iter_trajectory_responses(traj: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = traj.get("sample_id", "sample")
    responses: list[dict[str, Any]] = []
    groups = [traj.get("initial_responses", [])]
    groups.extend(traj.get("debate_rounds", []))
    for group in groups:
        for resp in group:
            item = dict(resp)
            round_num = int(item.get("round", 0))
            agent_id = int(item.get("agent_id", 0))
            item["sample_id"] = sample_id
            item["round"] = round_num
            item["agent_id"] = agent_id
            item["response_id"] = item.get("response_id") or make_response_id(
                sample_id,
                round_num,
                agent_id,
            )
            responses.append(item)
    return responses


def _normalize_answer(answer: object, task_type: str) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if task_type in {"multiple_choice", "mixed", "yes_no"}:
        return text.strip("().").upper()
    return text


def _is_correct(answer: object, correct_answer: object, task_type: str) -> bool:
    from src.algorithms.reward import math_answers_equal

    if not answer:
        return False
    if task_type == "math":
        return math_answers_equal(str(answer), str(correct_answer))
    return _normalize_answer(answer, task_type) == _normalize_answer(correct_answer, task_type)


def build_per_response_labels(traj: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = traj.get("sample_id", "sample")
    sample = traj.get("sample", {})
    task_type = sample.get("task_type", "multiple_choice")
    correct_answer = sample.get("answer", "")
    labels: list[dict[str, Any]] = []

    for resp in iter_trajectory_responses(traj):
        answer = resp.get("answer")
        labels.append({
            "sample_id": sample_id,
            "response_id": resp.get("response_id", ""),
            "round": resp.get("round", 0),
            "agent_id": resp.get("agent_id", 0),
            "agent_name": resp.get("agent_name", ""),
            "prompted_style": resp.get("prompted_style", ""),
            "source_split": resp.get("source_split") or traj.get("source_split", ""),
            "subject": resp.get("subject") or traj.get("subject", ""),
            "temperature": resp.get("temperature"),
            "task_type": task_type,
            "generation_index": resp.get("generation_index"),
            "response": resp.get("response", ""),
            "answer": answer,
            "is_correct": _is_correct(answer, correct_answer, task_type),
            "primary_style": None,
            "secondary_styles": [],
            "reasoning_style": None,
            "reasoning_style_confidence": 0.0,
            "format_status": None,
            "classification_source": None,
            "style_match": False,
            "trainable_for_actor": False,
            "style_verified": False,
            "style_weight": 0.0,
            "accepted_for_actor_sft": False,
        })
    return labels


def update_actor_acceptance(label: dict[str, Any], args: Any) -> None:
    """Populate actor SFT gate fields for one classified label."""
    prompted_style = str(label.get("prompted_style") or "")
    primary_style = str(label.get("primary_style") or "")
    task_type = str(label.get("task_type") or "multiple_choice")
    confidence = float(label.get("reasoning_style_confidence", 0.0) or 0.0)
    min_confidence = float(getattr(args, "min_style_confidence", 0.65))
    style_match = bool(prompted_style and primary_style == prompted_style)
    trainable = bool(label.get("is_correct")) and is_trainable_actor_response(
        label.get("response", ""),
        task_type,
    )
    style_verified = style_match and confidence >= min_confidence
    label["style_match"] = style_match
    label["trainable_for_actor"] = trainable
    label["style_verified"] = style_verified
    label["style_weight"] = confidence if style_match else 0.3
    label["accepted_for_actor_sft"] = (
        trainable
        and style_verified
        and bool(prompted_style)
        and primary_style == prompted_style
    )


def _choices_text(sample: dict[str, Any]) -> str:
    choices = sample.get("choices", "")
    if isinstance(choices, (list, dict)):
        return json.dumps(choices, ensure_ascii=False)
    return str(choices or "")


def classify_label(label: dict[str, Any], sample: dict[str, Any], args: Any) -> bool:
    question = sample.get("question", "")
    try:
        if label.get("is_correct"):
            result = classify_reasoning_style(
                response=label.get("response", ""),
                question=question,
                correct_answer=sample.get("answer", ""),
                use_api=True,
                cache_dir=args.output_dir,
                api_key=args.api_key,
                api_base=args.api_base,
                api_model=args.api_model,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            label["primary_style"] = result.primary_style.value
            label["reasoning_style"] = result.primary_style.value
            label["secondary_styles"] = [s.value for s in result.secondary_styles]
            label["reasoning_style_confidence"] = result.confidence
            label["format_status"] = result.format_status
            label["classification_source"] = "shared_classifier"
            update_actor_acceptance(label, args)
            return True

        label["classification_source"] = "local_correctness_only"
        update_actor_acceptance(label, args)
        return True
    except ClassificationError as e:
        logger.warning("Classification failed for %s: %s", label.get("response_id", ""), e)
        update_actor_acceptance(label, args)
        return False


def aggregate_result(
    sample_id: str,
    sample: dict[str, Any],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    style_labels = [
        (label.get("primary_style"), float(label.get("reasoning_style_confidence", 0.0)))
        for label in labels
        if label.get("is_correct") and label.get("primary_style")
    ]

    primary_style = None
    style_confidence = 0.0
    if style_labels:
        primary_style = Counter(style for style, _ in style_labels).most_common(1)[0][0]
        style_confidence = sum(
            conf for style, conf in style_labels if style == primary_style
        ) / max(1, sum(1 for style, _ in style_labels if style == primary_style))

    metadata = {
        "num_correct": sum(1 for label in labels if label.get("is_correct")),
        "num_incorrect": sum(1 for label in labels if not label.get("is_correct")),
        "num_format_failure": sum(
            1 for label in labels
            if label.get("format_status") in {"answer_only", "empty_or_invalid"}
        ),
        "num_responses": len(labels),
        "num_style_match": sum(1 for label in labels if label.get("style_match")),
        "num_trainable_for_actor": sum(
            1 for label in labels if label.get("trainable_for_actor")
        ),
        "num_style_verified": sum(
            1 for label in labels if label.get("style_verified")
        ),
        "num_accepted_for_actor_sft": sum(
            1 for label in labels if label.get("accepted_for_actor_sft")
        ),
        "classification_sources": dict(Counter(
            label.get("classification_source") or "unclassified"
            for label in labels
        )),
    }

    return {
        "sample_id": sample_id,
        "sample": sample,
        "question": sample.get("question", ""),
        "choices": _choices_text(sample),
        "primary_style": primary_style,
        "reasoning_style": primary_style,
        "reasoning_style_confidence": style_confidence,
        "metadata": metadata,
        "per_response_labels": labels,
    }


def classify_trajectory(traj: dict[str, Any], idx: int, args: Any) -> dict[str, Any]:
    sample_id = traj.get("sample_id", f"sample_{idx}")
    sample = dict(traj.get("sample", {}))
    sample.setdefault("dataset_name", getattr(args, "dataset", ""))
    labels = build_per_response_labels(traj)

    attempts = sum(1 for label in labels if label.get("is_correct"))
    failures = 0
    for label in labels:
        if not classify_label(label, sample, args):
            failures += 1

    return {
        "idx": idx,
        "sample_id": sample_id,
        "result": aggregate_result(sample_id, sample, labels),
        "attempts": attempts,
        "failures": failures,
    }


def classification_counts(results: list[dict[str, Any]]) -> tuple[int, int]:
    """Count style-classification attempts/failures across all saved results."""
    attempts = 0
    failures = 0
    for result in results:
        for label in result.get("per_response_labels", []):
            if not label.get("is_correct"):
                continue
            attempts += 1
            if label.get("classification_source") != "shared_classifier":
                failures += 1
    return attempts, failures


def build_reports(
    results: list[dict[str, Any]],
    min_style_confidence: float,
    api_failure_rate: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    style_dist: Counter[str] = Counter()
    format_dist: Counter[str] = Counter()
    per_subject: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {
        "styles": Counter(),
        "formats": Counter(),
    })

    total_labels = 0
    style_match_dist: Counter[str] = Counter()
    style_verified_dist: Counter[str] = Counter()
    trainable_style_dist: Counter[str] = Counter()
    accepted_sft_style_dist: Counter[str] = Counter()
    actor_sft_by_style: dict[str, Counter[str]] = defaultdict(Counter)
    actor_sft_by_subject_style: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prompted_classified: Counter[str] = Counter()
    gate_retention: dict[str, Counter[str]] = defaultdict(Counter)
    warnings: list[str] = []

    for result in results:
        sample = result.get("sample", {})
        subject = sample.get("subject", sample.get("category", "unknown"))

        for label in result.get("per_response_labels", []):
            total_labels += 1
            fmt = label.get("format_status") or "unclassified"
            format_dist[fmt] += 1
            per_subject[subject]["formats"][fmt] += 1
            prompted = str(label.get("prompted_style") or "unknown")
            gate_retention[prompted]["total"] += 1
            actor_sft_by_style[prompted]["generated"] += 1
            if label.get("answer") is not None:
                gate_retention[prompted]["parseable"] += 1
            if label.get("is_correct"):
                gate_retention[prompted]["correct"] += 1
                actor_sft_by_style[prompted]["correct"] += 1
            if label.get("trainable_for_actor"):
                gate_retention[prompted]["trainable"] += 1
            if label.get("style_match"):
                gate_retention[prompted]["style_match"] += 1
                if label.get("is_correct"):
                    actor_sft_by_style[prompted]["style_matched"] += 1
            if (
                float(label.get("reasoning_style_confidence", 0.0) or 0.0)
                >= min_style_confidence
            ):
                gate_retention[prompted]["confidence_pass"] += 1
            if label.get("style_verified"):
                gate_retention[prompted]["style_verified"] += 1
            if label.get("accepted_for_actor_sft"):
                gate_retention[prompted]["accepted_for_actor_sft"] += 1
                actor_sft_by_style[prompted]["usable"] += 1
                actor_sft_by_subject_style[subject][prompted] += 1
            if label.get("primary_style"):
                style = label["primary_style"]
                style_dist[style] += 1
                per_subject[subject]["styles"][style] += 1
                if label.get("style_match"):
                    style_match_dist[style] += 1
                if label.get("trainable_for_actor"):
                    trainable_style_dist[style] += 1
                if label.get("style_verified"):
                    style_verified_dist[style] += 1
                if label.get("accepted_for_actor_sft"):
                    accepted_sft_style_dist[style] += 1
                if prompted:
                    confusion[prompted][style] += 1
                    prompted_classified[prompted] += 1

    correct_style_total = sum(style_dist.values())
    if correct_style_total:
        direct_ratio = style_dist.get("direct", 0) / correct_style_total
        if direct_ratio > 0.80:
            warnings.append(f"direct style ratio {direct_ratio:.3f} exceeds 0.80")
        for style in sorted(VALID_STYLES):
            ratio = style_dist.get(style, 0) / correct_style_total
            if ratio < 0.05:
                warnings.append(f"style '{style}' ratio {ratio:.3f} is below 0.05")

    if total_labels:
        format_failure_ratio = (
            format_dist.get("answer_only", 0) + format_dist.get("empty_or_invalid", 0)
        ) / total_labels
        if format_failure_ratio > 0.20:
            warnings.append(f"format_failure ratio {format_failure_ratio:.3f} exceeds 0.20")

    confusion_matrix: dict[str, dict[str, int]] = {}
    for prompted in sorted(confusion):
        confusion_matrix[prompted] = dict(confusion[prompted])

    style_match_rate_by_prompted: dict[str, float] = {}
    for prompted in sorted(confusion):
        total = prompted_classified[prompted]
        matched = confusion[prompted].get(prompted, 0)
        style_match_rate_by_prompted[prompted] = round(matched / total, 4) if total else 0.0

    report = {
        "schema_version": 4,
        "classification_mode": "actor_sft_selection",
        "style_distribution": dict(style_dist),
        "style_match_distribution": dict(style_match_dist),
        "trainable_for_actor_distribution": dict(trainable_style_dist),
        "style_verified_distribution": dict(style_verified_dist),
        "accepted_for_actor_sft_distribution": dict(accepted_sft_style_dist),
        "format_status_distribution": dict(format_dist),
        "min_style_confidence": min_style_confidence,
        "prompted_vs_classified_confusion": confusion_matrix,
        "style_match_rate_by_prompted_style": style_match_rate_by_prompted,
        "gate_retention_by_prompted_style": {
            style: dict(counts)
            for style, counts in sorted(gate_retention.items())
        },
        "per_subject_distribution": {
            subject: {
                "styles": dict(counters["styles"]),
                "formats": dict(counters["formats"]),
            }
            for subject, counters in per_subject.items()
        },
        "warnings": warnings,
    }

    usable_counts = [
        actor_sft_by_style[style].get("usable", 0)
        for style in sorted(VALID_STYLES)
    ]
    max_usable = max(usable_counts) if usable_counts else 0
    min_usable = min(usable_counts) if usable_counts else 0
    style_imbalance_ratio = (
        round(max_usable / min_usable, 4)
        if min_usable > 0
        else None
    )
    actor_sft_report = {
        "schema_version": 4,
        "selection_mode": "actor_sft_candidates",
        "min_style_confidence": min_style_confidence,
        "by_style": {
            style: {
                "generated": actor_sft_by_style[style].get("generated", 0),
                "correct": actor_sft_by_style[style].get("correct", 0),
                "style_matched": actor_sft_by_style[style].get("style_matched", 0),
                "usable": actor_sft_by_style[style].get("usable", 0),
            }
            for style in sorted(VALID_STYLES)
        },
        "by_subject_style": {
            subject: {
                style: actor_sft_by_subject_style[subject].get(style, 0)
                for style in sorted(VALID_STYLES)
            }
            for subject in sorted(actor_sft_by_subject_style)
        },
        "subject_coverage": len(per_subject),
        "usable_subject_coverage": len(actor_sft_by_subject_style),
        "api_failure_rate": api_failure_rate,
        "style_imbalance_ratio": style_imbalance_ratio,
    }
    return report, actor_sft_report


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.api_key:
        logger.error(
            "API key is required. Set --api_key, GPT_API_KEY, "
            "OPENAI_API_KEY, or GLM_API_KEY."
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Classify Actor SFT Candidates (schema v4)")
    logger.info("  Input dir: %s", args.input_dir)
    logger.info("  Output dir: %s", args.output_dir)
    logger.info("  API provider: %s", getattr(args, "api_provider", ""))
    logger.info("  API model: %s", args.api_model)
    logger.info("=" * 60)

    trajectories = load_trajectories(args.input_dir)
    checkpoint = load_checkpoint(args.output_dir, args)
    completed_ids = set(checkpoint.get("completed", []))
    results = checkpoint.get("results", [])

    existing_by_id = {r.get("sample_id"): r for r in results}
    pending = [
        (idx, traj)
        for idx, traj in enumerate(trajectories)
        if traj.get("sample_id", f"sample_{idx}") not in completed_ids
    ]

    classification_attempts = 0
    classification_failures = 0
    max_workers = max(1, int(getattr(args, "max_workers", 4)))

    logger.info("Already classified: %s/%s", len(completed_ids), len(trajectories))
    logger.info("Classifying %s pending trajectories with %s workers", len(pending), max_workers)

    if max_workers == 1:
        processed = [classify_trajectory(traj, idx, args) for idx, traj in pending]
    else:
        processed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(classify_trajectory, traj, idx, args): idx
                for idx, traj in pending
            }
            for future in as_completed(future_map):
                processed.append(future.result())

    for count, item in enumerate(sorted(processed, key=lambda x: x["idx"]), start=1):
        existing_by_id[item["sample_id"]] = item["result"]
        completed_ids.add(item["sample_id"])
        classification_attempts += item["attempts"]
        classification_failures += item["failures"]
        if count % int(args.batch_size) == 0:
            save_checkpoint(args.output_dir, {
                "schema_version": 4,
                "completed": sorted(completed_ids),
                "results": list(existing_by_id.values()),
            }, args)
            logger.info(
                "Progress: %s/%s, failures=%s",
                len(completed_ids),
                len(trajectories),
                classification_failures,
            )

    results = list(existing_by_id.values())
    classification_attempts, classification_failures = classification_counts(results)
    save_checkpoint(args.output_dir, {
        "schema_version": 4,
        "completed": sorted(completed_ids),
        "results": results,
    }, args)

    failure_rate = (
        classification_failures / classification_attempts
        if classification_attempts else 0.0
    )
    strict = bool(getattr(args, "strict_classification", True))
    max_failure_rate = float(getattr(args, "max_classification_failure_rate", 0.0))
    if strict and failure_rate > max_failure_rate:
        raise RuntimeError(
            f"Classification failure rate {failure_rate:.3f} exceeds threshold "
            f"{max_failure_rate:.3f} ({classification_failures}/{classification_attempts})"
        )

    report, actor_sft_report = build_reports(
        results,
        min_style_confidence=float(getattr(args, "min_style_confidence", 0.65)),
        api_failure_rate=failure_rate,
    )

    output_file = Path(args.output_dir) / "classified_data.json"
    with open(output_file, "w") as f:
        json.dump({
            "schema_version": 4,
            "results": results,
            "metadata": {
                "classification_mode": "actor_sft_selection",
                "total_trajectories": len(trajectories),
                "api_model": args.api_model,
                "strict_classification": strict,
                "classification_attempts": classification_attempts,
                "classification_failures": classification_failures,
                "classification_failure_rate": failure_rate,
                "max_classification_failure_rate": max_failure_rate,
                "max_workers": max_workers,
                "min_style_confidence": float(getattr(args, "min_style_confidence", 0.65)),
            },
        }, f, indent=2, ensure_ascii=False)

    report_file = Path(args.output_dir) / "classification_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    actor_sft_report_file = Path(args.output_dir) / "actor_sft_candidate_report.json"
    with open(actor_sft_report_file, "w") as f:
        json.dump(actor_sft_report, f, indent=2, ensure_ascii=False)

    logger.info("Classification complete")
    logger.info("  Results: %s", output_file)
    logger.info("  Report: %s", report_file)
    logger.info("  Actor SFT candidate report: %s", actor_sft_report_file)
    logger.info("  Warnings: %s", report.get("warnings", []))


if __name__ == "__main__":
    main()
