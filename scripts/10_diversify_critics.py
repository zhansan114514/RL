"""
Train first-round Critics with supervised feedback data.

This phase runs after Actor SFT. It no longer builds Critic DPO
chosen/rejected pairs. Instead it constructs correction/keep SFT examples from
first-round multi-Actor outputs and trains one LoRA Critic per skill.

Usage:
    python scripts/10_diversify_critics.py \
        --config configs/society/experiment_mmlu.yaml
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.prompts.prompt_builder import build_simple_actor_prompt
from src.society.critic_sft_data import (
    attach_revision_result,
    build_actor_output,
    build_critic_sft_candidates,
    examples_to_sft_rows,
    sample_id_for,
    save_jsonl,
    select_examples_per_skill,
    summarize_candidates,
)
from src.society.multi_deliberation import LoRAError
from src.society.society_trainer import LoRAModelAdapter
from src.society.agent_registry import ReasoningStyle, resolve_reasoning_style
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)


STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen3-14B",
    "dataset": "mmlu",
    "input_dir": "output/society/classified",
    "actor_base_dir": "output/society/actors",
    "output_dir": "output/society/critics",
    "critic_skills": ["reasoning", "knowledge", "grounding", "verification"],
    "max_samples": 300,
    "max_examples_per_critic": 800,
    "min_examples_per_critic": 64,
    "correction_ratio": 0.55,
    "keep_ratio": 0.45,
    "summary_mode": "rule",
    "validate_revision": True,
    "fail_on_low_data": True,
    "actor_max_tokens": 192,
    "revision_max_tokens": 192,
    "summary_max_chars_per_actor": 500,
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
    "max_model_len": 4096,
    "max_lora_rank": 256,
    "mmlu_load_mode": "by_subject",
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Train first-round Critics with SFT",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/society/experiment_mmlu.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step04_diversify_critics", defaults=STEP_DEFAULTS).to_namespace()


def load_classified_data(input_dir: str) -> dict[str, Any]:
    """Load classified data produced by phase 2."""

    classified_file = os.path.join(input_dir, "classified_data.json")
    if not os.path.exists(classified_file):
        raise FileNotFoundError(f"Classified data not found: {classified_file}")
    with open(classified_file, encoding="utf-8") as f:
        return json.load(f)


def load_source_samples(input_dir: str, max_samples: int | None) -> list[dict[str, Any]]:
    """Load source samples from classified data with stable sample ids."""

    classified_data = load_classified_data(input_dir)
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, result in enumerate(classified_data.get("results", [])):
        sample = dict(result.get("sample") or {})
        if not sample:
            continue
        sid = str(result.get("sample_id") or sample_id_for(sample, idx))
        if sid in seen:
            continue
        seen.add(sid)
        sample["sample_id"] = sid
        if not str(sample.get("answer") or "").strip():
            continue
        samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError(
            f"No source samples with gold answers found in {input_dir}"
        )
    return samples


def load_actor_lora_paths(actor_dir: str) -> dict[str, str]:
    """Load Actor LoRA paths from the phase 3 actor registry."""

    registry_file = os.path.join(actor_dir, "actor_registry.json")
    if not os.path.exists(registry_file):
        logger.warning("Actor registry not found: %s", registry_file)
        return {}

    with open(registry_file, encoding="utf-8") as f:
        registry = json.load(f)

    paths: dict[str, str] = {}
    for style, info in registry.get("actors", {}).items():
        model_path = str(info.get("model_path") or "")
        if model_path:
            paths[style] = model_path
    logger.info("Loaded %s Actor LoRA paths from registry", len(paths))
    return paths


def build_actor_adapters(
    engine: Any,
    lora_paths: dict[str, str],
) -> dict[str, LoRAModelAdapter]:
    """Create one LoRA generation adapter per Actor."""

    from src.society.multi_deliberation import _load_lora_adapter

    adapters: dict[str, LoRAModelAdapter] = {}
    for raw_name, path in sorted(lora_paths.items()):
        actor_name, _ = actor_identity(raw_name)
        try:
            lora_req = _load_lora_adapter(engine, path)
        except LoRAError as e:
            raise LoRAError(
                f"Required LoRA adapter for Actor '{actor_name}' failed to "
                f"load from '{path}': {e}"
            ) from e
        if lora_req is None:
            raise LoRAError(
                f"Required LoRA adapter for Actor '{actor_name}' at '{path}' "
                "produced no LoRARequest."
            )
        adapters[actor_name] = LoRAModelAdapter(engine, lora_req)
        logger.info("Loaded LoRA for %s: %s", actor_name, path)
    return adapters


def actor_identity(raw_name: str) -> tuple[str, ReasoningStyle]:
    """Resolve registry keys like 'direct' or 'actor_direct' to Actor identity."""

    style_key = str(raw_name or "").strip()
    if style_key.startswith("actor_"):
        style_key = style_key.removeprefix("actor_")
    style = resolve_reasoning_style(style_key)
    return f"actor_{style.value}", style


def generate_initial_actor_outputs(
    *,
    samples: list[dict[str, Any]],
    actor_adapters: dict[str, LoRAModelAdapter],
    dataset_name: str,
    max_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    """Generate first-round responses for every Actor on every sample."""

    outputs: list[dict[str, Any]] = []
    for actor_name, adapter in sorted(actor_adapters.items()):
        _, style = actor_identity(actor_name)
        prompts = [
            build_simple_actor_prompt(
                sample,
                dataset_name,
                style=style,
                no_think=True,
            )
            for sample in samples
        ]
        logger.info(
            "Generating first-round outputs for %s (%s samples)",
            actor_name,
            len(prompts),
        )
        responses = adapter.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if len(responses) != len(samples):
            raise RuntimeError(
                f"Actor {actor_name} returned {len(responses)} responses for "
                f"{len(samples)} prompts"
            )
        for sample, response in zip(samples, responses):
            outputs.append(
                build_actor_output(
                    sample_id=sample_id_for(sample),
                    sample=sample,
                    actor_name=actor_name,
                    actor_style=style.value,
                    response=response,
                )
            )
    return outputs


def validate_with_actor_revision(
    *,
    examples: list[dict[str, Any]],
    actor_adapters: dict[str, LoRAModelAdapter],
    dataset_name: str,
    max_tokens: int,
    temperature: float,
    max_examples: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run Actor revision with candidate Critic feedback and keep successes."""

    if max_examples is not None and max_examples >= 0:
        examples_to_validate = examples[:max_examples]
    else:
        examples_to_validate = examples

    by_actor: dict[str, list[dict[str, Any]]] = {}
    for example in examples_to_validate:
        actor_name = str(example.get("metadata", {}).get("actor_name") or "")
        by_actor.setdefault(actor_name, []).append(example)

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()

    for actor_name, actor_examples in sorted(by_actor.items()):
        adapter = actor_adapters.get(actor_name)
        if adapter is None:
            for example in actor_examples:
                failed_case = dict(example)
                failed_case["validation_error"] = "missing_actor_adapter"
                failed.append(failed_case)
                failure_counts["missing_actor_adapter"] += 1
            continue
        _, style = actor_identity(actor_name)
        prompts = [
            _build_revision_prompt(
                example,
                dataset_name=dataset_name,
                style=style,
            )
            for example in actor_examples
        ]
        logger.info(
            "Validating Critic feedback through %s revisions (%s examples)",
            actor_name,
            len(prompts),
        )
        revised_responses = adapter.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if len(revised_responses) != len(actor_examples):
            raise RuntimeError(
                f"Actor {actor_name} returned {len(revised_responses)} revision "
                f"responses for {len(actor_examples)} prompts"
            )
        for example, revised_response in zip(actor_examples, revised_responses):
            updated, ok, reason = attach_revision_result(
                example,
                revised_response=revised_response,
            )
            if ok:
                passed.append(updated)
            else:
                updated["validation_error"] = reason
                failed.append(updated)
                failure_counts[reason] += 1

    metrics = {
        "validated_examples": len(examples_to_validate),
        "passed_examples": len(passed),
        "failed_examples": len(failed),
        "pass_rate": len(passed) / len(examples_to_validate) if examples_to_validate else 0.0,
        "failure_counts": dict(failure_counts),
    }
    return passed, failed, metrics


def train_critic_sft(
    *,
    model_name: str,
    rows: list[dict[str, Any]],
    critic_skill: str,
    output_dir: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    max_length: int,
    seed: int,
    device: int,
) -> str:
    """Train one Critic LoRA adapter with response-only SFT."""

    from datasets import Dataset
    from src.training.sft_trainer import train_sft
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)
    critic_output_dir = os.path.join(output_dir, f"critic_{critic_skill}")
    os.makedirs(critic_output_dir, exist_ok=True)
    sft_rows = examples_to_sft_rows(rows)
    dataset = Dataset.from_dict({
        "prompt": [row["prompt"] for row in sft_rows],
        "response": [row["response"] for row in sft_rows],
    })
    logger.info(
        "Training Critic-%s with SFT (%s examples)",
        critic_skill,
        len(sft_rows),
    )
    return train_sft(
        model_name_or_path=model_name,
        sft_dataset=dataset,
        output_dir=critic_output_dir,
        model_type=model_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        max_length=max_length,
        seed=seed,
        device=device,
    )


def write_critic_registry(
    *,
    output_dir: str,
    model_name: str,
    critic_skills: list[str],
    critic_paths: dict[str, str],
    critic_metrics: dict[str, dict[str, Any]],
    inactive_critics: dict[str, dict[str, Any]],
    validate_revision: bool,
) -> None:
    """Write phase 4 Critic registry consumed by later phases."""

    registry_critics: dict[str, dict[str, Any]] = {}
    for skill in critic_skills:
        path = critic_paths.get(skill, "")
        if path:
            registry_critics[skill] = {
                "critic_skill": skill,
                "model_path": path,
                "base_model": model_name,
                "training_method": "sft",
                "status": critic_metrics.get(skill, {}).get("status", "trained_specialist"),
                "participates": True,
                "base_model_only": False,
                "metrics": critic_metrics.get(skill, {}),
            }
        else:
            registry_critics[skill] = {
                "critic_skill": skill,
                "model_path": "",
                "base_model": model_name,
                "training_method": "sft",
                "status": "frozen_base",
                "participates": True,
                "base_model_only": True,
                "inactive_reason": inactive_critics.get(skill, {}),
                "metrics": critic_metrics.get(skill, {}),
            }

    registry_file = os.path.join(output_dir, "critic_registry.json")
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 5,
            "training_method": "sft",
            "critics": registry_critics,
            "metadata": {
                "base_model": model_name,
                "training_method": "sft",
                "case_types": ["correction", "keep"],
                "validate_revision": validate_revision,
                "num_critics": len(registry_critics),
                "num_active_critics": len(critic_paths),
                "inactive_critics": inactive_critics,
            },
        }, f, indent=2, ensure_ascii=False)
    logger.info("Critic registry saved: %s", registry_file)


def _build_revision_prompt(
    example: dict[str, Any],
    *,
    dataset_name: str,
    style: ReasoningStyle,
) -> str:
    metadata = example.get("metadata", {})
    sample = dict(metadata.get("task") or {})
    return build_simple_actor_prompt(
        sample,
        dataset_name,
        round_num=1,
        previous_actor_response=str(metadata.get("actor_current_response") or ""),
        critic_feedback=str(example.get("response") or ""),
        style=style,
        no_think=True,
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _cleanup_gpu() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Train first-round Critics with SFT")
    logger.info("  Model: %s", args.model_name)
    logger.info("  Dataset: %s", args.dataset)
    logger.info("  Input dir: %s", args.input_dir)
    logger.info("  Actor dir: %s", args.actor_base_dir)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Critic skills: %s", args.critic_skills)
    logger.info("=" * 60)

    if getattr(args, "summary_mode", "rule") != "rule":
        raise ValueError("Only summary_mode='rule' is supported in this implementation")

    samples = load_source_samples(
        args.input_dir,
        max_samples=int(getattr(args, "max_samples", 0) or 0) or None,
    )
    logger.info("Loaded %s source samples", len(samples))

    actor_lora_paths = load_actor_lora_paths(args.actor_base_dir)
    if not actor_lora_paths:
        raise LoRAError(
            f"No Actor LoRA paths found in '{args.actor_base_dir}'. Run Actor "
            "SFT first and provide a valid actor_registry.json."
        )

    from src.inference.vllm_server import VLLMInference

    shared_engine = None
    actor_outputs: list[dict[str, Any]] = []
    selected_by_skill: dict[str, list[dict[str, Any]]] = {}
    failed_cases: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    try:
        shared_engine = VLLMInference(
            args.model_name,
            cuda_device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enable_lora=True,
            max_loras=max(1, len(actor_lora_paths)),
            max_lora_rank=args.max_lora_rank,
        )
        actor_adapters = build_actor_adapters(shared_engine, actor_lora_paths)
        actor_outputs = generate_initial_actor_outputs(
            samples=samples,
            actor_adapters=actor_adapters,
            dataset_name=args.dataset,
            max_tokens=int(args.actor_max_tokens),
            temperature=float(getattr(args, "actor_temperature", 0.7)),
        )

        candidate_examples, candidate_metrics = build_critic_sft_candidates(
            samples=samples,
            actor_outputs=actor_outputs,
            critic_skills=args.critic_skills,
            dataset_name=args.dataset,
            summary_max_chars_per_actor=int(args.summary_max_chars_per_actor),
        )
        metrics["actor_outputs"] = _summarize_actor_outputs(actor_outputs)
        metrics["candidate_metrics"] = candidate_metrics
        logger.info(
            "Built %s Critic SFT candidates: %s",
            len(candidate_examples),
            candidate_metrics.get("case_type_counts", {}),
        )

        if bool(getattr(args, "validate_revision", True)):
            max_validation = getattr(args, "max_revision_validation_examples", None)
            if max_validation is not None:
                max_validation = int(max_validation)
            passed_examples, failed_cases, validation_metrics = validate_with_actor_revision(
                examples=candidate_examples,
                actor_adapters=actor_adapters,
                dataset_name=args.dataset,
                max_tokens=int(args.revision_max_tokens),
                temperature=float(getattr(args, "actor_temperature", 0.7)),
                max_examples=max_validation,
            )
            trainable_examples = passed_examples
            metrics["revision_validation"] = validation_metrics
            logger.info(
                "Revision validation kept %s/%s examples",
                len(passed_examples),
                validation_metrics.get("validated_examples", 0),
            )
        else:
            trainable_examples = candidate_examples
            metrics["revision_validation"] = {"enabled": False}

        selected_by_skill, selection_metrics = select_examples_per_skill(
            trainable_examples,
            critic_skills=args.critic_skills,
            max_examples_per_critic=int(args.max_examples_per_critic),
            correction_ratio=float(args.correction_ratio),
            keep_ratio=float(args.keep_ratio),
            seed=int(args.seed),
        )
        metrics["selection_metrics"] = selection_metrics

    finally:
        try:
            del actor_adapters
        except UnboundLocalError:
            pass
        if shared_engine is not None:
            del shared_engine
            _cleanup_gpu()

    all_selected = [
        example
        for skill in args.critic_skills
        for example in selected_by_skill.get(skill, [])
    ]
    save_jsonl(all_selected, os.path.join(output_dir, "critic_sft_all.jsonl"))
    save_jsonl(failed_cases, os.path.join(output_dir, "critic_sft_failed_cases.jsonl"))

    critic_paths: dict[str, str] = {}
    critic_metrics: dict[str, dict[str, Any]] = {}
    inactive_critics: dict[str, dict[str, Any]] = {}

    for skill in args.critic_skills:
        rows = selected_by_skill.get(skill, [])
        skill_dir = os.path.join(output_dir, f"critic_{skill}")
        os.makedirs(skill_dir, exist_ok=True)
        save_jsonl(rows, os.path.join(skill_dir, "critic_sft_train.jsonl"))
        skill_metrics = summarize_candidates(rows)
        skill_metrics.update(metrics.get("selection_metrics", {}).get(skill, {}))
        critic_metrics[skill] = skill_metrics

        min_examples = int(args.min_examples_per_critic)
        if len(rows) < min_examples:
            inactive_critics[skill] = {
                "reason": "sft_examples_below_min_examples_per_critic",
                "num_examples": len(rows),
                "min_examples_per_critic": min_examples,
            }
            logger.warning(
                "Critic-%s has only %s examples; registering frozen_base",
                skill,
                len(rows),
            )
            continue

    if inactive_critics and bool(getattr(args, "fail_on_low_data", True)):
        metrics["critic_metrics"] = critic_metrics
        metrics["inactive_critics"] = inactive_critics
        _write_json(os.path.join(output_dir, "critic_sft_metrics.json"), metrics)
        raise RuntimeError(
            "One or more Critics have too few SFT examples: "
            f"{inactive_critics}"
        )

    for skill in args.critic_skills:
        rows = selected_by_skill.get(skill, [])
        if skill in inactive_critics:
            continue
        checkpoint_path = train_critic_sft(
            model_name=args.model_name,
            rows=rows,
            critic_skill=skill,
            output_dir=output_dir,
            lora_r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            learning_rate=float(args.learning_rate),
            batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            num_epochs=int(args.num_epochs),
            max_length=int(args.max_length),
            seed=int(args.seed),
            device=int(args.device),
        )
        critic_paths[skill] = checkpoint_path
        critic_metrics[skill]["status"] = "trained_specialist"

    metrics["critic_metrics"] = critic_metrics
    metrics["inactive_critics"] = inactive_critics
    _write_json(os.path.join(output_dir, "critic_sft_metrics.json"), metrics)

    write_critic_registry(
        output_dir=output_dir,
        model_name=args.model_name,
        critic_skills=list(args.critic_skills),
        critic_paths=critic_paths,
        critic_metrics=critic_metrics,
        inactive_critics=inactive_critics,
        validate_revision=bool(getattr(args, "validate_revision", True)),
    )

    logger.info("=" * 60)
    logger.info("Critic SFT training complete")
    logger.info("  Trained LoRA critics: %s", len(critic_paths))
    for skill, path in critic_paths.items():
        logger.info("    %s: %s", skill, path)
    logger.info("=" * 60)


def _summarize_actor_outputs(actor_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(actor_outputs)
    parseable = sum(1 for output in actor_outputs if output.get("actor_answer"))
    actor_counts = Counter(str(output.get("actor_name") or "unknown") for output in actor_outputs)
    actor_correct = Counter()
    actor_total = Counter()
    for output in actor_outputs:
        name = str(output.get("actor_name") or "unknown")
        actor_total[name] += 1
        if output.get("actor_correct"):
            actor_correct[name] += 1
    return {
        "total": total,
        "parseable": parseable,
        "parse_success_rate": parseable / total if total else 0.0,
        "actor_counts": dict(actor_counts),
        "actor_accuracy": {
            name: actor_correct[name] / count if count else 0.0
            for name, count in actor_total.items()
        },
    }


if __name__ == "__main__":
    main()
