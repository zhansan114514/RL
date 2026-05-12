"""
Diversify Actors by training one LoRA adapter per MMLU-aware reasoning style.
"""

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
from src.society.augmentation import (
    generate_style_conditioned_responses,
    make_synthetic_rejected_response,
)
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
    "min_pairs_per_actor": 256,
    "target_pairs_per_actor": 512,
    "max_pairs_per_actor": 1024,
    "max_pairs_per_sample": 4,
    "augment_when_below": 256,
    "max_synthetic_ratio": 0.7,
    "pair_mix": {"correctness": 0.7, "style": 0.3},
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "beta": 0.1,
    "seed": 42,
    "device": 0,
}

ACTOR_TRAINING_STYLE_GUIDANCE = {
    ReasoningStyle.DIRECT: (
        "Keep the reasoning short and only include what is needed to justify the answer."
    ),
    ReasoningStyle.EVIDENCE: (
        "Ground the reasoning in key facts, definitions, wording, or evidence from the problem."
    ),
    ReasoningStyle.ELIMINATION: (
        "Compare options and rule out weaker or incorrect choices before making the final decision."
    ),
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Diversify Actors")
    parser.add_argument("--config", type=str, default="configs/society/experiment_mmlu.yaml")
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step03_diversify_actors", defaults=STEP_DEFAULTS).to_namespace()
    if not getattr(args, "api_key", ""):
        args.api_key = os.environ.get("GLM_API_KEY", "")
    return args


def load_classified_data(input_dir: str) -> dict[str, Any]:
    path = Path(input_dir) / "classified_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Classified data not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version") != 3:
        raise ValueError(f"Expected classified_data schema_version=3, got {data.get('schema_version')}")
    logger.info("Loaded %s classified samples", len(data.get("results", [])))
    return data


def build_response_index(
    classified_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in classified_results:
        sample_id = result.get("sample_id", "")
        sample = result.get("sample", {})
        for label in result.get("per_response_labels", []):
            response_text = label.get("response", "")
            by_sample[sample_id].append({
                "sample_id": sample_id,
                "response_id": label.get("response_id", ""),
                "sample": sample,
                "response": response_text,
                "is_correct": bool(label.get("is_correct")),
                "primary_style": label.get("primary_style") or label.get("reasoning_style"),
                "secondary_styles": label.get("secondary_styles", []),
                "style_confidence": label.get("reasoning_style_confidence", 0.0),
                "prompted_style": label.get("prompted_style", ""),
                "style_match": bool(label.get("style_match")),
                "accepted_for_actor": bool(label.get("accepted_for_actor")),
                "generation_index": label.get("generation_index"),
                "agent_name": label.get("agent_name", ""),
                "synthetic": False,
            })
    return by_sample


def _take_limited(
    items: list[dict[str, Any]],
    limit_by_sample: Counter[str],
    max_pairs_per_sample: int,
    quota: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for item in items:
        if quota is not None and len(selected) >= quota:
            break
        sid = item.get("sample_id", "")
        if limit_by_sample[sid] >= max_pairs_per_sample:
            continue
        selected.append(item)
        limit_by_sample[sid] += 1
    return selected


def build_preference_pairs_for_style(
    by_sample: dict[str, list[dict[str, Any]]],
    target_style: str,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mix = dict(getattr(args, "pair_mix", {}) or {})
    target_pairs = int(getattr(args, "target_pairs_per_actor", 512))
    max_pairs = int(getattr(args, "max_pairs_per_actor", 1024))
    max_pairs_per_sample = int(getattr(args, "max_pairs_per_sample", 4))
    quotas = _pair_quotas(target_pairs, mix)

    correctness_candidates = []
    style_candidates = []

    for sample_id, responses in by_sample.items():
        style_correct = [
            r for r in responses
            if (
                r["is_correct"]
                and r.get("accepted_for_actor")
                and r.get("prompted_style") == target_style
                and r.get("primary_style") == target_style
            )
        ]
        style_correct_ids = {r.get("response_id", "") for r in style_correct}
        other_style_correct = [
            r for r in responses
            if (
                r["is_correct"]
                and r.get("response_id")
                and r.get("response_id") not in style_correct_ids
                and (
                    r.get("prompted_style") != target_style
                    or r.get("primary_style") != target_style
                    or not r.get("accepted_for_actor")
                )
            )
        ]
        incorrect = [r for r in responses if not r["is_correct"]]

        for chosen in style_correct:
            for rejected in incorrect:
                if rejected["response_id"] == chosen["response_id"]:
                    continue
                correctness_candidates.append(_make_pair(chosen, rejected, "correctness"))
            for rejected in other_style_correct:
                if rejected["response_id"] == chosen["response_id"]:
                    continue
                style_candidates.append(_make_pair(chosen, rejected, "style"))

    pairs: list[dict[str, Any]] = []
    per_sample_counter: Counter[str] = Counter()
    for kind, pool in (
        ("correctness", correctness_candidates),
        ("style", style_candidates),
    ):
        selected = _take_limited(
            pool,
            per_sample_counter,
            max_pairs_per_sample,
            quota=quotas[kind],
        )
        pairs.extend(selected)

    if len(pairs) < target_pairs:
        filler = correctness_candidates + style_candidates
        existing = {
            (
                p["metadata"].get("chosen_response_id"),
                p["metadata"].get("rejected_response_id"),
                p["metadata"].get("pair_type"),
            )
            for p in pairs
        }
        for pair in filler:
            key = (
                pair["metadata"].get("chosen_response_id"),
                pair["metadata"].get("rejected_response_id"),
                pair["metadata"].get("pair_type"),
            )
            if key in existing:
                continue
            sid = pair["metadata"].get("sample_id", "")
            if per_sample_counter[sid] >= max_pairs_per_sample:
                continue
            pairs.append(pair)
            per_sample_counter[sid] += 1
            existing.add(key)
            if len(pairs) >= target_pairs:
                break

    pairs = pairs[:max_pairs]
    metrics = _pair_metrics(pairs)
    metrics.update({
        "real_pair_count": len(pairs),
        "synthetic_pair_count": 0,
        "synthetic_ratio": 0.0,
        "candidate_counts": {
            "correctness": len(correctness_candidates),
            "style": len(style_candidates),
        },
    })
    return pairs, metrics


def augment_pairs_if_needed(
    pairs: list[dict[str, Any]],
    by_sample: dict[str, list[dict[str, Any]]],
    target_style: str,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_pairs = int(getattr(args, "min_pairs_per_actor", 256))
    augment_when_below = int(getattr(args, "augment_when_below", min_pairs))
    if len(pairs) >= augment_when_below:
        return pairs, {"synthetic_pair_count": 0, "synthetic_ratio": 0.0}

    max_synth_ratio = float(getattr(args, "max_synthetic_ratio", 0.7))
    max_total = int(getattr(args, "max_pairs_per_actor", 1024))
    needed = max(0, min_pairs - len(pairs))
    max_synth_by_ratio = int((max_synth_ratio * max(len(pairs), 1)) / max(1.0 - max_synth_ratio, 1e-6))
    max_synthetic = min(needed, max_synth_by_ratio, max_total - len(pairs))
    if max_synthetic <= 0:
        return pairs, {"synthetic_pair_count": 0, "synthetic_ratio": 0.0}

    samples = [items[0]["sample"] for items in by_sample.values() if items]
    synthetic = generate_style_conditioned_responses(
        samples=samples,
        target_style=target_style,
        dataset_name=args.dataset,
        max_generations=max_synthetic,
        api_key=getattr(args, "api_key", ""),
        api_base=getattr(args, "api_base", "https://api.labforge.top"),
        api_model=getattr(args, "api_model", "gpt5.5"),
        cache_dir=args.input_dir,
        style_confidence_threshold=float(getattr(args, "min_style_confidence", 0.65)),
        request_timeout=getattr(args, "request_timeout", 60),
        max_retries=getattr(args, "max_retries", 5),
        retry_delay=getattr(args, "retry_delay", 5),
    )

    synthetic_pairs = []
    for idx, item in enumerate(synthetic):
        sample = item["sample"]
        rejected = make_synthetic_rejected_response(sample)
        synthetic_pairs.append({
            "sample": sample,
            "chosen": item["response"],
            "rejected": rejected,
            "metadata": {
                "thinking_style": target_style,
                "sample_id": sample.get("question", f"synthetic_{idx}"),
                "chosen_response_id": f"synthetic_{target_style}_{idx}",
                "rejected_response_id": f"synthetic_rejected_{target_style}_{idx}",
                "pair_type": "synthetic_correctness",
                "synthetic": True,
                "chosen_prompted_style": target_style,
                "chosen_primary_style": target_style,
                "chosen_accepted_for_actor": True,
                "style_confidence": item.get("style_confidence", 0.0),
            },
        })

    pairs = (pairs + synthetic_pairs)[:max_total]
    synth_count = sum(1 for p in pairs if p.get("metadata", {}).get("synthetic"))
    return pairs, {
        "synthetic_pair_count": synth_count,
        "synthetic_ratio": synth_count / len(pairs) if pairs else 0.0,
    }


def _pair_quotas(target_pairs: int, mix: dict[str, float]) -> dict[str, int]:
    defaults = {"correctness": 0.7, "style": 0.3}
    ratios = {k: float(mix.get(k, defaults[k])) for k in defaults}
    total = sum(ratios.values()) or 1.0
    quotas = {k: int(round(target_pairs * ratios[k] / total)) for k in ratios}
    delta = target_pairs - sum(quotas.values())
    quotas["correctness"] += delta
    return quotas


def _make_pair(chosen: dict[str, Any], rejected: dict[str, Any], pair_type: str) -> dict[str, Any]:
    return {
        "sample": chosen["sample"],
        "chosen": chosen["response"],
        "rejected": rejected["response"],
        "metadata": {
            "thinking_style": chosen.get("primary_style"),
            "sample_id": chosen.get("sample_id", ""),
            "chosen_response_id": chosen.get("response_id", ""),
            "rejected_response_id": rejected.get("response_id", ""),
            "pair_type": pair_type,
            "synthetic": bool(chosen.get("synthetic") or rejected.get("synthetic")),
            "chosen_prompted_style": chosen.get("prompted_style", ""),
            "chosen_primary_style": chosen.get("primary_style", ""),
            "chosen_style_confidence": chosen.get("style_confidence", 0.0),
            "rejected_prompted_style": rejected.get("prompted_style", ""),
            "rejected_primary_style": rejected.get("primary_style", ""),
        },
    }


def _pair_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(pairs),
        "pair_type_counts": dict(Counter(
            p.get("metadata", {}).get("pair_type", "unknown")
            for p in pairs
        )),
        "unique_samples": len({
            p.get("metadata", {}).get("sample_id", "")
            for p in pairs
        }),
    }


def train_actor_dpo(
    model_name: str,
    preference_pairs: list[dict[str, Any]],
    thinking_style: str,
    output_dir: str,
    dataset_name: str,
    args: Any,
) -> str:
    from datasets import Dataset
    from src.training.dpo_trainer import train_dpo
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)
    actor_output_dir = os.path.join(output_dir, f"actor_{thinking_style}")
    os.makedirs(actor_output_dir, exist_ok=True)

    prompts = [
        _actor_training_prompt(dataset_name, thinking_style, p["sample"])
        for p in preference_pairs
    ]
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": [p["chosen"] for p in preference_pairs],
        "rejected": [p["rejected"] for p in preference_pairs],
    })
    return train_dpo(
        model_name_or_path=model_name,
        preference_dataset=dataset,
        output_dir=actor_output_dir,
        model_type=model_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
    )


def _actor_training_prompt(dataset_name: str, thinking_style: str, sample: dict[str, Any]) -> str:
    from src.society.agent_registry import ACTOR_STYLE_PROMPTS

    style = ReasoningStyle(thinking_style)
    base_prompt = build_simple_actor_prompt(sample, dataset_name, style=style)
    return (
        "/no_think\n"
        f"You are Actor-{thinking_style}.\n"
        "Use this reasoning style naturally.\n"
        f"{ACTOR_STYLE_PROMPTS[style]}\n"
        f"{ACTOR_TRAINING_STYLE_GUIDANCE[style]}\n\n"
        f"{base_prompt}\n\n"
        f"{_style_output_format(style)}"
    )


def _style_output_format(style: ReasoningStyle) -> str:
    if style in {
        ReasoningStyle.DIRECT,
        ReasoningStyle.EVIDENCE,
        ReasoningStyle.ELIMINATION,
    }:
        return (
            "Reason naturally in the requested style.\n"
            "At the end, write one final answer sentence:\n"
            "The final result is <answer>."
        )
    raise ValueError(f"Unsupported reasoning style: {style}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diversify Actors")
    logger.info("  Model: %s", args.model_name)
    logger.info("  Styles: %s", args.reasoning_styles)
    logger.info("=" * 60)

    classified = load_classified_data(args.input_dir)
    by_sample = build_response_index(classified["results"])

    actor_paths: dict[str, str] = {}
    actor_metrics: dict[str, Any] = {}

    for thinking_style in args.reasoning_styles:
        ReasoningStyle(thinking_style)
        logger.info("--- Building Actor pairs: %s ---", thinking_style)
        cache_path = Path(args.output_dir) / f"pairs_actor3_prompted_{thinking_style}.json"
        metrics_path = Path(args.output_dir) / f"pairs_actor3_prompted_{thinking_style}_metrics.json"

        pairs, metrics = build_preference_pairs_for_style(by_sample, thinking_style, args)
        pairs, synth_metrics = augment_pairs_if_needed(pairs, by_sample, thinking_style, args)
        metrics.update(synth_metrics)
        metrics["real_pair_count"] = len(pairs) - metrics.get("synthetic_pair_count", 0)
        metrics["trained_low_data"] = len(pairs) < int(args.min_pairs_per_actor)
        with open(cache_path, "w") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        actor_metrics[thinking_style] = metrics
        if len(pairs) < int(args.min_pairs_per_actor):
            raise RuntimeError(
                f"Actor {thinking_style} has {len(pairs)} pairs, below "
                f"min_pairs_per_actor={args.min_pairs_per_actor}"
            )

        checkpoint = train_actor_dpo(
            model_name=args.model_name,
            preference_pairs=pairs,
            thinking_style=thinking_style,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            args=args,
        )
        actor_paths[thinking_style] = checkpoint

    registry_file = Path(args.output_dir) / "actor_registry.json"
    with open(registry_file, "w") as f:
        json.dump({
            "schema_version": 3,
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
                "min_pairs_per_actor": args.min_pairs_per_actor,
                "target_pairs_per_actor": args.target_pairs_per_actor,
                "max_pairs_per_actor": args.max_pairs_per_actor,
            },
        }, f, indent=2, ensure_ascii=False)

    logger.info("Actor diversification complete: %s", registry_file)


if __name__ == "__main__":
    main()
