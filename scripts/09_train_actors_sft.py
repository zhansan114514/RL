"""Train one first-round Actor SFT LoRA adapter per reasoning style."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.prompts.prompt_builder import build_simple_actor_prompt
from src.society.agent_registry import ReasoningStyle
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen3-14B",
    "dataset": "mmlu",
    "cache_dir": "output/society",
    "input_dir": "output/society/classified",
    "output_dir": "output/society/actors",
    "reasoning_styles": ["direct", "evidence", "elimination"],
    "min_style_confidence": 0.55,
    "min_examples_per_actor": 128,
    "max_examples_per_question_style": 2,
    "balance_by_subject": True,
    "min_subject_coverage": 55,
    "max_style_imbalance_ratio": 1.5,
    "max_subject_imbalance_ratio": 3.0,
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    "device": 0,
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train Actor SFT adapters")
    parser.add_argument("--config", type=str, default="configs/society/experiment_mmlu.yaml")
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step03_train_actors_sft", defaults=STEP_DEFAULTS).to_namespace()


def load_classified_data(input_dir: str) -> dict[str, Any]:
    path = Path(input_dir) / "classified_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Classified data not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if int(data.get("schema_version", 0)) < 4:
        raise ValueError(
            f"Expected classified_data schema_version>=4 for Actor SFT, "
            f"got {data.get('schema_version')}"
        )
    logger.info("Loaded %s classified samples", len(data.get("results", [])))
    return data


def _actor_training_prompt(dataset_name: str, thinking_style: str, sample: dict[str, Any]) -> str:
    style = ReasoningStyle(thinking_style)
    return build_simple_actor_prompt(
        sample,
        dataset_name,
        style=style,
        no_think=True,
    )


def build_response_index(classified_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in classified_results:
        sample_id = result.get("sample_id", "")
        sample = result.get("sample", {})
        subject = sample.get("subject", sample.get("category", "unknown")) or "unknown"
        for label in result.get("per_response_labels", []):
            prompted_style = str(label.get("prompted_style") or "")
            if not prompted_style:
                continue
            by_style[prompted_style].append({
                "sample_id": sample_id,
                "response_id": label.get("response_id", ""),
                "sample": sample,
                "subject": label.get("subject") or subject,
                "source_split": label.get("source_split") or sample.get("source_split", ""),
                "question": sample.get("question", ""),
                "response": label.get("response", ""),
                "answer": label.get("answer", ""),
                "is_correct": bool(label.get("is_correct")),
                "primary_style": label.get("primary_style") or label.get("reasoning_style"),
                "style_confidence": float(label.get("reasoning_style_confidence", 0.0) or 0.0),
                "prompted_style": prompted_style,
                "style_match": bool(label.get("style_match")),
                "trainable_for_actor": bool(label.get("trainable_for_actor")),
                "style_verified": bool(label.get("style_verified")),
                "accepted_for_actor_sft": bool(label.get("accepted_for_actor_sft")),
                "temperature": label.get("temperature"),
                "generation_index": label.get("generation_index"),
            })
    return by_style


def _dedupe_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        key = (str(example.get("question") or ""), str(example.get("response") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def _cap_examples_per_question_style(
    examples: list[dict[str, Any]],
    max_examples_per_question_style: int,
) -> list[dict[str, Any]]:
    if max_examples_per_question_style <= 0:
        return examples
    selected = []
    counts: Counter[str] = Counter()
    for example in examples:
        question = str(example.get("question") or example.get("sample_id") or "")
        if counts[question] >= max_examples_per_question_style:
            continue
        selected.append(example)
        counts[question] += 1
    return selected


def _balance_by_subject_loose(
    examples: list[dict[str, Any]],
    max_subject_imbalance_ratio: float,
) -> list[dict[str, Any]]:
    if not examples or max_subject_imbalance_ratio <= 0:
        return examples

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_subject[str(example.get("subject") or "unknown")].append(example)

    non_empty_counts = [len(items) for items in by_subject.values() if items]
    if not non_empty_counts:
        return examples
    avg_count = len(examples) / max(1, len(by_subject))
    max_per_subject = max(1, int(round(avg_count * max_subject_imbalance_ratio)))

    balanced = []
    for subject in sorted(by_subject):
        balanced.extend(by_subject[subject][:max_per_subject])
    return balanced


def select_sft_examples_for_style(
    by_style: dict[str, list[dict[str, Any]]],
    target_style: str,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_confidence = float(getattr(args, "min_style_confidence", 0.55))
    candidates = [
        item for item in by_style.get(target_style, [])
        if (
            item.get("accepted_for_actor_sft")
            and item.get("prompted_style") == target_style
            and item.get("primary_style") == target_style
            and float(item.get("style_confidence", 0.0) or 0.0) >= min_confidence
        )
    ]
    candidates = sorted(
        candidates,
        key=lambda item: (
            str(item.get("sample_id") or ""),
            -float(item.get("style_confidence", 0.0) or 0.0),
            str(item.get("response_id") or ""),
        ),
    )
    deduped = _dedupe_examples(candidates)
    capped = _cap_examples_per_question_style(
        deduped,
        int(getattr(args, "max_examples_per_question_style", 2)),
    )
    if bool(getattr(args, "balance_by_subject", True)):
        selected = _balance_by_subject_loose(
            capped,
            float(getattr(args, "max_subject_imbalance_ratio", 3.0)),
        )
    else:
        selected = capped

    subject_counts = Counter(str(item.get("subject") or "unknown") for item in selected)
    confidence_values = [
        float(item.get("style_confidence", 0.0) or 0.0)
        for item in selected
    ]
    metrics = {
        "training_examples": len(selected),
        "candidate_examples": len(candidates),
        "deduped_examples": len(deduped),
        "after_question_cap": len(capped),
        "subject_coverage": len(subject_counts),
        "subject_counts": dict(sorted(subject_counts.items())),
        "style_confidence_mean": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else 0.0
        ),
        "min_style_confidence": min_confidence,
        "max_examples_per_question_style": int(
            getattr(args, "max_examples_per_question_style", 2)
        ),
        "balance_by_subject": bool(getattr(args, "balance_by_subject", True)),
    }
    return selected, metrics


def examples_to_sft_dataset(
    examples: list[dict[str, Any]],
    dataset_name: str,
    target_style: str,
) -> list[dict[str, Any]]:
    rows = []
    for item in examples:
        sample = item.get("sample", {})
        rows.append({
            "prompt": _actor_training_prompt(dataset_name, target_style, sample),
            "response": item.get("response", ""),
            "metadata": {
                "sample_id": item.get("sample_id", ""),
                "response_id": item.get("response_id", ""),
                "source_split": item.get("source_split", ""),
                "subject": item.get("subject", ""),
                "style": target_style,
                "prompted_style": item.get("prompted_style", ""),
                "primary_style": item.get("primary_style", ""),
                "temperature": item.get("temperature"),
                "generation_index": item.get("generation_index"),
                "style_confidence": item.get("style_confidence", 0.0),
                "answer": item.get("answer", ""),
            },
        })
    return rows


def validate_global_balance(actor_metrics: dict[str, dict[str, Any]], args: Any) -> None:
    counts = [
        int(metrics.get("training_examples", 0))
        for metrics in actor_metrics.values()
    ]
    if counts and min(counts) > 0:
        ratio = max(counts) / min(counts)
        max_ratio = float(getattr(args, "max_style_imbalance_ratio", 1.5))
        if ratio > max_ratio:
            raise RuntimeError(
                f"Actor SFT style imbalance ratio {ratio:.3f} exceeds "
                f"max_style_imbalance_ratio={max_ratio:.3f}"
            )


def train_actor_sft(
    model_name: str,
    rows: list[dict[str, Any]],
    thinking_style: str,
    output_dir: str,
    args: Any,
) -> str:
    from datasets import Dataset
    from src.training.sft_trainer import train_sft
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)
    actor_output_dir = os.path.join(output_dir, f"actor_{thinking_style}")
    os.makedirs(actor_output_dir, exist_ok=True)
    dataset = Dataset.from_dict({
        "prompt": [row["prompt"] for row in rows],
        "response": [row["response"] for row in rows],
    })
    return train_sft(
        model_name_or_path=model_name,
        sft_dataset=dataset,
        output_dir=actor_output_dir,
        model_type=model_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Train Actors With SFT")
    logger.info("  Model: %s", args.model_name)
    logger.info("  Styles: %s", args.reasoning_styles)
    logger.info("  Input dir: %s", args.input_dir)
    logger.info("  Output dir: %s", args.output_dir)
    logger.info("=" * 60)

    classified = load_classified_data(args.input_dir)
    by_style = build_response_index(classified["results"])

    actor_paths: dict[str, str] = {}
    actor_metrics: dict[str, Any] = {}
    actor_rows: dict[str, list[dict[str, Any]]] = {}

    for thinking_style in args.reasoning_styles:
        ReasoningStyle(thinking_style)
        logger.info("--- Selecting Actor SFT examples: %s ---", thinking_style)
        examples, metrics = select_sft_examples_for_style(by_style, thinking_style, args)
        min_examples = int(getattr(args, "min_examples_per_actor", 128))
        min_subject_coverage = int(getattr(args, "min_subject_coverage", 0) or 0)
        if len(examples) < min_examples:
            raise RuntimeError(
                f"Actor {thinking_style} has {len(examples)} SFT examples, below "
                f"min_examples_per_actor={min_examples}"
            )
        if metrics["subject_coverage"] < min_subject_coverage:
            raise RuntimeError(
                f"Actor {thinking_style} has subject coverage "
                f"{metrics['subject_coverage']}, below "
                f"min_subject_coverage={min_subject_coverage}"
            )

        rows = examples_to_sft_dataset(examples, args.dataset, thinking_style)
        dataset_path = Path(args.output_dir) / f"sft_actor_{thinking_style}.json"
        metrics_path = Path(args.output_dir) / f"sft_actor_{thinking_style}_metrics.json"
        with open(dataset_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(
            "  Selected %s examples across %s subjects",
            len(rows),
            metrics["subject_coverage"],
        )
        actor_rows[thinking_style] = rows
        actor_metrics[thinking_style] = metrics

    validate_global_balance(actor_metrics, args)

    for thinking_style, rows in actor_rows.items():
        logger.info("--- Training Actor SFT: %s ---", thinking_style)
        checkpoint = train_actor_sft(
            model_name=args.model_name,
            rows=rows,
            thinking_style=thinking_style,
            output_dir=args.output_dir,
            args=args,
        )
        actor_paths[thinking_style] = checkpoint

    registry_file = Path(args.output_dir) / "actor_registry.json"
    with open(registry_file, "w") as f:
        json.dump({
            "schema_version": 4,
            "training_method": "sft",
            "actors": {
                style: {
                    "thinking_style": style,
                    "model_path": path,
                    "base_model": args.model_name,
                    "metrics": actor_metrics.get(style, {}),
                }
                for style, path in actor_paths.items()
            },
            "metadata": {
                "base_model": args.model_name,
                "num_actors": len(actor_paths),
                "training_method": "sft",
                "min_examples_per_actor": args.min_examples_per_actor,
                "min_subject_coverage": args.min_subject_coverage,
                "max_style_imbalance_ratio": args.max_style_imbalance_ratio,
                "max_subject_imbalance_ratio": args.max_subject_imbalance_ratio,
            },
        }, f, indent=2, ensure_ascii=False)

    logger.info("Actor SFT training complete: %s", registry_file)


if __name__ == "__main__":
    main()
