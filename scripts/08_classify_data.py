"""
Classify bootstrap data with the shared MMLU-aware society classifier.

Outputs schema_version=3 classified_data.json and classification_report.json.
The script is intentionally a caller of src.society.data_classifier; it does
not maintain its own label taxonomy.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.society.agent_registry import ReasoningStyle
from src.society.data_classifier import (
    ERROR_PROFILE_DIMENSIONS,
    ERROR_PROFILE_PRIMARY_LABELS,
    ClassificationError,
    classify_error_profile,
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
}

VALID_STYLES = {style.value for style in ReasoningStyle}
VALID_PRIMARY = set(ERROR_PROFILE_PRIMARY_LABELS)


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


def load_trajectories(input_dir: str) -> list[dict]:
    import glob

    pattern = os.path.join(input_dir, "trajectories.jsonl")
    if not os.path.exists(pattern):
        files = glob.glob(os.path.join(input_dir, "*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No trajectories found in {input_dir}")
        pattern = files[0]

    trajectories = []
    with open(pattern) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    logger.info("Loaded %s trajectories from %s", len(trajectories), pattern)
    return trajectories


def load_checkpoint(output_dir: str) -> dict[str, Any]:
    path = Path(output_dir) / "classification_checkpoint.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"schema_version": 3, "completed": [], "results": []}


def save_checkpoint(output_dir: str, checkpoint: dict[str, Any]) -> None:
    path = Path(output_dir) / "classification_checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "response": resp.get("response", ""),
            "answer": answer,
            "is_correct": _is_correct(answer, correct_answer, task_type),
            "primary_style": None,
            "secondary_styles": [],
            "reasoning_style": None,
            "reasoning_style_confidence": 0.0,
            "format_status": None,
            "error_profile": None,
            "classification_source": None,
        })
    return labels


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
            return True

        result = classify_error_profile(
            response=label.get("response", ""),
            question=question,
            extracted_answer=label.get("answer") or "",
            correct_answer=sample.get("answer", ""),
            choices=sample.get("choices", ""),
            dataset_name=sample.get("dataset_name", ""),
            task_type=sample.get("task_type", ""),
            subject=sample.get("subject", sample.get("category", "")),
            use_api=True,
            cache_dir=args.output_dir,
            api_key=args.api_key,
            api_base=args.api_base,
            api_model=args.api_model,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        label["format_status"] = result.format_status
        label["error_profile"] = {
            "format_status": result.format_status,
            "scores": result.scores,
            "primary": result.primary,
            "secondary": result.secondary,
            "confidence": result.confidence,
            "evidence": result.evidence,
        }
        label["classification_source"] = "shared_classifier"
        return True
    except ClassificationError as e:
        logger.warning("Classification failed for %s: %s", label.get("response_id", ""), e)
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
    error_profiles = [
        label.get("error_profile")
        for label in labels
        if not label.get("is_correct") and label.get("error_profile")
    ]

    primary_style = None
    style_confidence = 0.0
    if style_labels:
        primary_style = Counter(style for style, _ in style_labels).most_common(1)[0][0]
        style_confidence = sum(
            conf for style, conf in style_labels if style == primary_style
        ) / max(1, sum(1 for style, _ in style_labels if style == primary_style))

    error_profile = None
    if error_profiles:
        avg_scores = {
            dim: sum(p.get("scores", {}).get(dim, 0.0) for p in error_profiles) / len(error_profiles)
            for dim in ERROR_PROFILE_DIMENSIONS
        }
        primary = Counter(p.get("primary", "ambiguous") for p in error_profiles).most_common(1)[0][0]
        error_profile = {
            "format_status": Counter(
                p.get("format_status", "valid") for p in error_profiles
            ).most_common(1)[0][0],
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
        "num_correct": sum(1 for label in labels if label.get("is_correct")),
        "num_incorrect": sum(1 for label in labels if not label.get("is_correct")),
        "num_format_failure": sum(
            1 for label in labels
            if label.get("format_status") in {"answer_only", "empty_or_invalid"}
            or (label.get("error_profile") or {}).get("primary") == "format_failure"
        ),
        "num_responses": len(labels),
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
        "error_profile": error_profile,
        "metadata": metadata,
        "per_response_labels": labels,
    }


def classify_trajectory(traj: dict[str, Any], idx: int, args: Any) -> dict[str, Any]:
    sample_id = traj.get("sample_id", f"sample_{idx}")
    sample = dict(traj.get("sample", {}))
    sample.setdefault("dataset_name", getattr(args, "dataset", ""))
    labels = build_per_response_labels(traj)

    attempts = len(labels)
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


def build_reports(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    style_splits: dict[str, list[str]] = defaultdict(list)
    profile_splits: dict[str, list[str]] = defaultdict(list)
    per_response_style_splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_response_profile_splits: dict[str, list[dict[str, Any]]] = defaultdict(list)

    style_dist: Counter[str] = Counter()
    profile_dist: Counter[str] = Counter()
    format_dist: Counter[str] = Counter()
    per_subject: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {
        "styles": Counter(),
        "errors": Counter(),
        "formats": Counter(),
    })

    total_labels = 0
    warnings: list[str] = []

    for result in results:
        sample_id = result["sample_id"]
        sample = result.get("sample", {})
        subject = sample.get("subject", sample.get("category", "unknown"))

        if result.get("primary_style"):
            style_splits[result["primary_style"]].append(sample_id)
        profile = result.get("error_profile")
        if profile and profile.get("primary") in VALID_PRIMARY:
            profile_splits[profile["primary"]].append(sample_id)

        for label in result.get("per_response_labels", []):
            total_labels += 1
            fmt = label.get("format_status") or "unclassified"
            format_dist[fmt] += 1
            per_subject[subject]["formats"][fmt] += 1
            if label.get("primary_style"):
                style = label["primary_style"]
                style_dist[style] += 1
                per_subject[subject]["styles"][style] += 1
                per_response_style_splits[style].append({
                    "sample_id": sample_id,
                    "response_id": label.get("response_id", ""),
                    "round": label.get("round", 0),
                    "agent_id": label.get("agent_id", 0),
                    "agent_name": label.get("agent_name", ""),
                    "is_correct": label.get("is_correct", False),
                })
            profile = label.get("error_profile")
            if profile and profile.get("primary") in VALID_PRIMARY:
                primary = profile["primary"]
                profile_dist[primary] += 1
                per_subject[subject]["errors"][primary] += 1
                per_response_profile_splits[primary].append({
                    "sample_id": sample_id,
                    "response_id": label.get("response_id", ""),
                    "round": label.get("round", 0),
                    "agent_id": label.get("agent_id", 0),
                    "agent_name": label.get("agent_name", ""),
                    "is_correct": label.get("is_correct", False),
                    "scores": profile.get("scores", {}),
                    "confidence": profile.get("confidence", 0.0),
                })

    correct_style_total = sum(style_dist.values())
    if correct_style_total:
        direct_ratio = style_dist.get("direct", 0) / correct_style_total
        if direct_ratio > 0.55:
            warnings.append(f"direct style ratio {direct_ratio:.3f} exceeds 0.55")
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
    profile_total = sum(profile_dist.values())
    if profile_total:
        ambiguous_ratio = profile_dist.get("ambiguous", 0) / profile_total
        if ambiguous_ratio > 0.20:
            warnings.append(f"ambiguous error ratio {ambiguous_ratio:.3f} exceeds 0.20")

    splits = {
        "reasoning_styles": dict(style_splits),
        "error_profiles": dict(profile_splits),
        "per_response_reasoning_styles": dict(per_response_style_splits),
        "per_response_error_profiles": dict(per_response_profile_splits),
    }
    report = {
        "schema_version": 3,
        "style_distribution": dict(style_dist),
        "error_profile_distribution": dict(profile_dist),
        "format_status_distribution": dict(format_dist),
        "per_subject_distribution": {
            subject: {
                "styles": dict(counters["styles"]),
                "errors": dict(counters["errors"]),
                "formats": dict(counters["formats"]),
            }
            for subject, counters in per_subject.items()
        },
        "warnings": warnings,
    }
    return splits, report


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
    logger.info("Classify Bootstrap Data (schema v3)")
    logger.info("  Input dir: %s", args.input_dir)
    logger.info("  Output dir: %s", args.output_dir)
    logger.info("  API provider: %s", getattr(args, "api_provider", ""))
    logger.info("  API model: %s", args.api_model)
    logger.info("=" * 60)

    trajectories = load_trajectories(args.input_dir)
    checkpoint = load_checkpoint(args.output_dir)
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
                "schema_version": 3,
                "completed": sorted(completed_ids),
                "results": list(existing_by_id.values()),
            })
            logger.info(
                "Progress: %s/%s, failures=%s",
                len(completed_ids),
                len(trajectories),
                classification_failures,
            )

    results = list(existing_by_id.values())
    save_checkpoint(args.output_dir, {
        "schema_version": 3,
        "completed": sorted(completed_ids),
        "results": results,
    })

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

    splits, report = build_reports(results)

    output_file = Path(args.output_dir) / "classified_data.json"
    with open(output_file, "w") as f:
        json.dump({
            "schema_version": 3,
            "results": results,
            "metadata": {
                "total_trajectories": len(trajectories),
                "api_model": args.api_model,
                "strict_classification": strict,
                "classification_attempts": classification_attempts,
                "classification_failures": classification_failures,
                "classification_failure_rate": failure_rate,
                "max_classification_failure_rate": max_failure_rate,
                "max_workers": max_workers,
            },
        }, f, indent=2, ensure_ascii=False)

    splits_file = Path(args.output_dir) / "splits.json"
    with open(splits_file, "w") as f:
        json.dump({
            "reasoning_styles": splits["reasoning_styles"],
            "error_profiles": splits["error_profiles"],
        }, f, indent=2, ensure_ascii=False)

    per_response_splits_file = Path(args.output_dir) / "per_response_splits.json"
    with open(per_response_splits_file, "w") as f:
        json.dump({
            "reasoning_styles": splits["per_response_reasoning_styles"],
            "error_profiles": splits["per_response_error_profiles"],
        }, f, indent=2, ensure_ascii=False)

    report_file = Path(args.output_dir) / "classification_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Classification complete")
    logger.info("  Results: %s", output_file)
    logger.info("  Report: %s", report_file)
    logger.info("  Warnings: %s", report.get("warnings", []))


if __name__ == "__main__":
    main()
