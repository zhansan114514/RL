"""
Society evaluation and ablation experiments.

Runs A1-A5 ablation experiments and computes:
- Per-round accuracy
- Improvement rate
- Consensus accuracy
- Diversity metrics
- Wilson 95% CI

Key design: loads the base model ONCE, all ablation experiments share
the same vLLM engine and switch LoRA adapters dynamically.

Usage:
    python scripts/12_society_evaluate.py \
        --config configs/society/experiment_mmlu.yaml \
        --run_ablations
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen3-14B",
    "dataset": "math",
    "cache_dir": "output/society",
    "society_dir": "output/society/society",
    "output_dir": "output/society/eval",
    "num_rounds": 2,
    "max_tokens": 1024,
    "temperature": 0.0,
    "evaluation_mode": "single_gpu",
    "evaluation_modes": {
        "single_gpu": {
            "devices": [0],
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.65,
        },
        "dual_gpu": {
            "devices": [0, 1],
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.80,
        },
    },
    "dtype": "bfloat16",
    "run_ablations": True,
    "num_samples_for_qualitative": 5,
    "max_model_len": 4096,
    "max_lora_rank": 256,
    "router_top_k": 2,
    "router_min_confidence": 0.1,
    "router_fallback_to_uniform": False,
    "route_feedback_to_actor": True,
    "consensus_uses_selected_critics_only": False,
    "max_samples": None,
    "sampling": None,
    "mmlu_load_mode": "by_subject",
    "eval_batch_size": 1,
}


ROUTER_CONFIG_FALLBACKS = {
    "router_top_k": ("top_k", STEP_DEFAULTS["router_top_k"]),
    "router_min_confidence": ("min_confidence", STEP_DEFAULTS["router_min_confidence"]),
    "router_fallback_to_uniform": (
        "fallback_to_uniform",
        STEP_DEFAULTS["router_fallback_to_uniform"],
    ),
    "route_feedback_to_actor": (
        "route_feedback_to_actor",
        STEP_DEFAULTS["route_feedback_to_actor"],
    ),
    "consensus_uses_selected_critics_only": (
        "consensus_uses_selected_critics_only",
        STEP_DEFAULTS["consensus_uses_selected_critics_only"],
    ),
}


@dataclass
class EvalResult:
    """Result of society evaluation."""
    initial_accuracy: float
    final_consensus_accuracy: float
    per_round_accuracy: List[float]
    relative_improvement: float
    absolute_improvement: float
    mean_actor_final_accuracy: float
    best_actor_oracle_accuracy: float
    diversity_score: float
    ci_95: tuple[float, float]
    num_samples: int
    eval_time_seconds: float
    sample_details: List[Dict[str, Any]]
    parsing_diagnostics: Dict[str, Any] = field(default_factory=dict)
    deliberation_dynamics: Dict[str, int] = field(default_factory=dict)
    per_round_consensus_confidence: List[float] = field(default_factory=list)
    critic_metrics: Dict[str, Any] = field(default_factory=dict)
    ablation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRuntime:
    """Resolved GPU runtime settings for evaluation."""
    mode: str
    devices: List[int]
    tensor_parallel_size: int
    gpu_memory_utilization: float


def resolve_evaluation_runtime(args) -> EvaluationRuntime:
    """Resolve the selected evaluation GPU mode from step06_evaluate config."""
    mode = args.evaluation_mode
    modes = args.evaluation_modes

    if not isinstance(modes, dict) or mode not in modes:
        available = sorted(modes) if isinstance(modes, dict) else []
        raise ValueError(
            f"Unknown evaluation_mode '{mode}'. "
            f"Available modes: {available}"
        )

    mode_cfg = modes[mode] or {}
    devices = mode_cfg.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError(
            f"evaluation_modes.{mode}.devices must be a non-empty list of GPU ids"
        )
    devices = [int(device) for device in devices]

    tensor_parallel_size = int(
        mode_cfg.get("tensor_parallel_size") or len(devices)
    )
    if tensor_parallel_size != len(devices):
        raise ValueError(
            f"evaluation_modes.{mode}.tensor_parallel_size must match number of "
            f"devices for this evaluation mode: tensor_parallel_size="
            f"{tensor_parallel_size}, devices={devices}"
        )

    gpu_memory_utilization = float(mode_cfg.get("gpu_memory_utilization"))
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError(
            f"evaluation_modes.{mode}.gpu_memory_utilization must be in (0, 1], "
            f"got {gpu_memory_utilization}"
        )

    return EvaluationRuntime(
        mode=mode,
        devices=devices,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _apply_router_config_fallbacks(args, router_cfg: Dict[str, Any]) -> None:
    """Fill evaluation router args from top-level router config when step fields are absent."""
    router_cfg = router_cfg if isinstance(router_cfg, dict) else {}
    for arg_name, (router_key, default) in ROUTER_CONFIG_FALLBACKS.items():
        if not hasattr(args, arg_name):
            setattr(args, arg_name, router_cfg.get(router_key, default))


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Society evaluation",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_mmlu.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--no_ablations", action="store_true",
        help="Skip ablation experiments.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step(
        "step06_evaluate",
        defaults={
            key: value
            for key, value in STEP_DEFAULTS.items()
            if key not in ROUTER_CONFIG_FALLBACKS
        },
    ).to_namespace()
    _apply_router_config_fallbacks(args, cfg.section("router"))

    # Preserve config path for logging
    args.config = cli_args.config

    if cli_args.no_ablations:
        args.run_ablations = False

    return args


def load_society_registry(society_dir: str) -> Dict[str, Any]:
    """Load society agent registry."""
    registry_file = os.path.join(society_dir, "final_agent_registry.json")

    if not os.path.exists(society_dir):
        logger.warning(f"Society directory not found: {society_dir}")
        return {}

    if not os.path.exists(registry_file):
        logger.warning(f"Registry not found: {registry_file}")
        return {}

    with open(registry_file) as f:
        return json.load(f)


def load_eval_dataset(dataset_name: str, seed: int, sampling: Optional[dict] = None, mmlu_load_mode: str = "by_subject") -> List[Dict]:
    """Load evaluation dataset (test split)."""
    from src.data.loader import load_dataset

    data = load_dataset(
        dataset_name,
        seed=seed,
        sampling=sampling,
        mmlu_load_mode=mmlu_load_mode,
    )
    test_data = data.get("test", [])

    if not test_data:
        raise ValueError(
            f"Test split is empty for dataset={dataset_name}. "
            f"Check data loading configuration."
        )

    return test_data


def compute_diversity(responses: List[str]) -> float:
    """Compute diversity score using unique response ratio."""
    if not responses:
        return 0.0
    return len(set(responses)) / len(responses)


def _compute_ci(
    predictions: List[Optional[str]],
    labels: List[str],
    task_types: List[str],
) -> tuple:
    """Compute mixed-task accuracy and Wilson confidence interval margin."""
    from src.evaluation.answer_resolution import compute_accuracy_with_ci_mixed

    return compute_accuracy_with_ci_mixed(predictions, labels, task_types)


# ============================================================
# Agent config helpers
# ============================================================

def _build_agent_configs(
    registry: Dict[str, Any],
    actor_names: Optional[List[str]] = None,
    critic_names: Optional[List[str]] = None,
):
    """Build AgentConfig lists and LoRA paths from registry.

    Returns (actor_configs, critic_configs, lora_paths).
    """
    from src.society.agent_registry import (
        AgentConfig, AgentRole, resolve_reasoning_style, resolve_critic_skill,
    )

    actors_info = registry.get("actors", {})
    critics_info = registry.get("critics", {})
    base_model = registry.get("training_config", {}).get("base_model", "Qwen/Qwen3-14B")

    # Filter by names if specified
    if actor_names is not None:
        actors_info = {k: v for k, v in actors_info.items() if k in actor_names}
    if critic_names is not None:
        critics_info = {k: v for k, v in critics_info.items() if k in critic_names}

    actor_configs = []
    for name, info in actors_info.items():
        style_str = name.replace("actor_", "")
        try:
            style = resolve_reasoning_style(style_str)
        except ValueError as e:
            logger.error(f"Cannot resolve actor style '{style_str}': {e}")
            raise
        actor_configs.append(AgentConfig(
            name=name,
            role=AgentRole.ACTOR,
            reasoning_style=style,
            model_path=base_model,
            lora_path=info.get("model_path", ""),
            system_prompt="",
        ))

    critic_configs = []
    for name, info in critics_info.items():
        skill_str = name.replace("critic_", "")
        try:
            skill = resolve_critic_skill(skill_str)
        except ValueError as e:
            logger.error(f"Cannot resolve critic skill '{skill_str}': {e}")
            raise
        critic_configs.append(AgentConfig(
            name=name,
            role=AgentRole.CRITIC,
            error_specialty=skill,
            model_path=base_model,
            lora_path=info.get("model_path", ""),
            system_prompt="",
        ))

    lora_paths = {}
    for name, info in actors_info.items():
        path = info.get("model_path", "")
        if path:
            lora_paths[name] = path
    for name, info in critics_info.items():
        path = info.get("model_path", "")
        if path:
            lora_paths[name] = path

    return actor_configs, critic_configs, lora_paths


def _build_base_agent_configs(model_name: str):
    """Build generic AgentConfigs WITHOUT LoRA for a true base-model baseline.

    This represents the untrained base model (no LoRA at all), acting as a
    single Actor and single Critic.  Used as the zero-training reference
    point — distinct from A1 which uses trained LoRA adapters.
    """
    from src.society.agent_registry import AgentConfig, AgentRole, ReasoningStyle, CriticSkill

    actor_config = AgentConfig(
        name="base_actor",
        role=AgentRole.ACTOR,
        reasoning_style=ReasoningStyle.DIRECT,
        model_path=model_name,
        lora_path="",  # No LoRA — pure base model
        system_prompt="",
    )
    critic_config = AgentConfig(
        name="base_critic",
        role=AgentRole.CRITIC,
        error_specialty=CriticSkill.REASONING,
        model_path=model_name,
        lora_path="",  # No LoRA — pure base model
        system_prompt="",
    )
    return [actor_config], [critic_config], {}


def _build_agent_configs_from_phase_registries(
    actor_phase_dir: str,
    critic_phase_dir: str,
    base_model: str,
    actor_names: Optional[List[str]] = None,
    critic_names: Optional[List[str]] = None,
):
    """Build AgentConfigs from phase 3/4 diversification registries.

    These are PRE-society-training LoRA adapters, used for ablation A1-A3
    to isolate diversification-only effects from society training effects.

    Args:
        actor_phase_dir: Directory containing actor_registry.json (from script 09)
        critic_phase_dir: Directory containing critic_registry.json (from script 10)
        base_model: Base model path
        actor_names: Optional filter for specific actor style names
        critic_names: Optional filter for specific critic skill names
    """
    from src.society.agent_registry import (
        AgentConfig, AgentRole,
        resolve_reasoning_style, resolve_critic_skill,
    )

    actor_configs = []
    critic_configs = []
    lora_paths = {}

    # Load actor registry
    actor_reg_file = os.path.join(actor_phase_dir, "actor_registry.json")
    if os.path.exists(actor_reg_file):
        with open(actor_reg_file) as f:
            actor_data = json.load(f)

        for style_key, info in actor_data.get("actors", {}).items():
            if actor_names is not None and style_key not in actor_names:
                continue
            try:
                style = resolve_reasoning_style(style_key)
            except ValueError as e:
                logger.error(f"Cannot resolve phase actor style '{style_key}': {e}")
                raise

            path = info.get("model_path", "")
            name = f"actor_{style.value}"
            actor_configs.append(AgentConfig(
                name=name,
                role=AgentRole.ACTOR,
                reasoning_style=style,
                model_path=base_model,
                lora_path=path,
                system_prompt="",
            ))
            if path:
                lora_paths[name] = path
    else:
        logger.warning(f"Phase actor registry not found: {actor_reg_file}")

    # Load critic registry
    critic_reg_file = os.path.join(critic_phase_dir, "critic_registry.json")
    if os.path.exists(critic_reg_file):
        with open(critic_reg_file) as f:
            critic_data = json.load(f)

        for skill_key, info in critic_data.get("critics", {}).items():
            if critic_names is not None and skill_key not in critic_names:
                continue
            try:
                skill = resolve_critic_skill(skill_key)
            except ValueError as e:
                logger.error(f"Cannot resolve phase critic skill '{skill_key}': {e}")
                raise

            path = info.get("model_path", "")
            name = f"critic_{skill.value}"
            critic_configs.append(AgentConfig(
                name=name,
                role=AgentRole.CRITIC,
                error_specialty=skill,
                model_path=base_model,
                lora_path=path,
                system_prompt="",
            ))
            if path:
                lora_paths[name] = path
    else:
        logger.warning(f"Phase critic registry not found: {critic_reg_file}")

    return actor_configs, critic_configs, lora_paths


def _run_deliberation_on_samples(
    engine,
    actor_configs,
    critic_configs,
    samples: List[Dict],
    dataset_name: str,
    lora_paths: Dict[str, str],
    num_rounds: int,
    max_tokens: int,
    temperature: float,
    router_top_k: int = 2,
    router_uniform: bool = False,
    router_min_confidence: float = 0.1,
    router_fallback_to_uniform: bool = False,
    route_feedback_to_actor: bool = True,
    consensus_uses_selected_critics_only: bool = False,
    eval_batch_size: int = 1,
) -> EvalResult:
    """Run deliberation on samples with a shared vLLM engine. No model loading."""
    from src.society.multi_deliberation import multi_agent_deliberate_single_gpu
    from src.society.multi_deliberation import multi_agent_deliberate_batched_single_gpu
    from src.society.router import CriticRouter

    # Create router with the specified configuration
    router = CriticRouter(
        top_k=router_top_k,
        min_confidence=router_min_confidence,
        fallback_to_uniform=router_fallback_to_uniform,
        uniform_weights=router_uniform,
    )

    start_time = time.time()
    from src.evaluation.answer_resolution import (
        answers_match,
        source_rates,
    )

    labels = [s.get("answer", "") for s in samples]
    task_types = [s.get("task_type", "yes_no") for s in samples]
    per_round_answers: Dict[int, List[Optional[str]]] = {
        r: [] for r in range(num_rounds)
    }
    per_round_confidences: Dict[int, List[float]] = {r: [] for r in range(num_rounds)}
    actor_sources_by_round: Dict[int, List[str]] = {r: [] for r in range(num_rounds)}
    critic_confidence_parsed = 0
    critic_suggested_parsed = 0
    critic_answer_correct_parsed = 0
    critic_total = 0
    critic_selected_total = 0
    critic_parse_errors: Counter[str] = Counter()
    all_final_answers: List[Optional[str]] = []
    all_initial_answers: List[Optional[str]] = []
    final_actor_correct_counts = [0 for _ in samples]
    final_actor_total_counts = [0 for _ in samples]
    final_actor_any_correct = [False for _ in samples]
    details = []

    for si, sample in enumerate(samples):
        if (si + 1) % 5 == 0 or si == 0:
            logger.info(f"    Sample {si + 1}/{len(samples)}")

    # ---- Choose single-sample or batched mode ----
    if eval_batch_size > 1 and len(samples) > 1:
        # Batched mode: process samples in chunks for higher GPU utilisation
        logger.info(
            f"  Using batched deliberation (batch_size={eval_batch_size})"
        )
        all_results: List[Any] = [None] * len(samples)
        for batch_start in range(0, len(samples), eval_batch_size):
            batch = samples[batch_start:batch_start + eval_batch_size]
            batch_results = multi_agent_deliberate_batched_single_gpu(
                inference_engine=engine,
                actors=actor_configs,
                critics=critic_configs,
                samples=batch,
                dataset_name=dataset_name,
                lora_paths=lora_paths,
                num_rounds=num_rounds,
                max_tokens=max_tokens,
                temperature=temperature,
                router=router,
                route_feedback_to_actor=route_feedback_to_actor,
                consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
            )
            for j, result in enumerate(batch_results):
                all_results[batch_start + j] = result
            logger.info(
                f"    Batch {batch_start + 1}-{batch_start + len(batch)}"
                f"/{len(samples)} done"
            )
    else:
        # Single-sample mode (backward compatible)
        all_results = []
        for sample in samples:
            result = multi_agent_deliberate_single_gpu(
                inference_engine=engine,
                actors=actor_configs,
                critics=critic_configs,
                sample=sample,
                dataset_name=dataset_name,
                lora_paths=lora_paths,
                num_rounds=num_rounds,
                max_tokens=max_tokens,
                temperature=temperature,
                router=router,
                route_feedback_to_actor=route_feedback_to_actor,
                consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
            )
            all_results.append(result)

    # ---- Aggregate results ----
    for si, (sample, result) in enumerate(zip(samples, all_results)):
        final_answer = result.consensus_answer
        all_final_answers.append(final_answer)

        for rnd in result.rounds:
            if rnd.round_num in per_round_answers:
                per_round_answers[rnd.round_num].append(rnd.consensus_answer)
                per_round_confidences[rnd.round_num].append(rnd.consensus_confidence)
                actor_sources_by_round[rnd.round_num].extend(
                    rnd.actor_answer_sources.values()
                )

            routed_feedbacks = getattr(rnd, "routed_feedbacks", {}) or {}
            critic_feedbacks = getattr(rnd, "critic_feedbacks", {}) or {}
            selected_by_actor = {
                actor_name: {
                    getattr(fb, "critic_name", "")
                    for fb in selected_feedbacks
                }
                for actor_name, selected_feedbacks in routed_feedbacks.items()
                if isinstance(selected_feedbacks, list)
            }
            for actor_name, feedbacks in critic_feedbacks.items():
                selected = selected_by_actor.get(actor_name, set())
                critic_selected_total += len(selected)
                for feedback in feedbacks.values():
                    critic_total += 1
                    critic_confidence_parsed += int(feedback.confidence is not None)
                    critic_suggested_parsed += int(feedback.suggested_answer is not None)
                    critic_answer_correct_parsed += int(feedback.answer_correct != "unknown")
                    critic_parse_errors.update(feedback.parse_errors)

        # Compute initial answer as majority vote across all actors (not just first)
        initial_answer = None
        if result.rounds:
            first_round = result.rounds[0]
            init_answers = [a for a in first_round.actor_answers.values() if a is not None]
            if init_answers:
                counter = Counter(init_answers)
                initial_answer = counter.most_common(1)[0][0]
        all_initial_answers.append(initial_answer)

        final_actor_answers = result.final_answers or {}
        for answer in final_actor_answers.values():
            final_actor_total_counts[si] += 1
            if answers_match(answer, sample.get("answer", ""), sample.get("task_type", "yes_no")):
                final_actor_correct_counts[si] += 1
                final_actor_any_correct[si] = True

        initially_correct = answers_match(
            initial_answer,
            sample.get("answer", ""),
            sample.get("task_type", "yes_no"),
        )
        finally_correct = answers_match(
            final_answer,
            sample.get("answer", ""),
            sample.get("task_type", "yes_no"),
        )

        details.append({
            "question": sample.get("question", ""),
            "initial_answer": initial_answer,
            "final_answer": final_answer,
            "confidence": result.consensus_confidence,
            "ground_truth": sample.get("answer", ""),
            "task_type": sample.get("task_type", "yes_no"),
            "initially_correct": initially_correct,
            "finally_correct": finally_correct,
            "flipped_to_correct": not initially_correct and finally_correct,
            "flipped_to_wrong": initially_correct and not finally_correct,
            "actor_final_answers": final_actor_answers,
            "rounds": [
                {
                    "round_num": rnd.round_num,
                    "consensus_answer": rnd.consensus_answer,
                    "consensus_confidence": rnd.consensus_confidence,
                    "actor_answers": {
                        name: {
                            "resolved": rnd.actor_answers.get(name),
                            "source": rnd.actor_answer_sources.get(name, "none"),
                            "parse_confidence": getattr(
                                rnd,
                                "actor_parse_confidence",
                                {},
                            ).get(name, 0.0),
                        }
                        for name in rnd.actor_responses
                    },
                }
                for rnd in result.rounds
            ],
        })

    initial_acc, _ = _compute_ci(all_initial_answers, labels, task_types)
    final_acc, ci_margin = _compute_ci(all_final_answers, labels, task_types)
    ci_95 = (max(0, final_acc - ci_margin), min(1, final_acc + ci_margin))

    round_accs = []
    per_round_consensus_confidence = []
    for r in range(num_rounds):
        if per_round_answers[r]:
            acc, _ = _compute_ci(
                per_round_answers[r],
                labels[:len(per_round_answers[r])],
                task_types[:len(per_round_answers[r])],
            )
            round_accs.append(acc)
            confidences = per_round_confidences[r]
            per_round_consensus_confidence.append(
                sum(confidences) / len(confidences) if confidences else 0.0
            )

    final_actor_total = sum(final_actor_total_counts)
    mean_actor_final_accuracy = (
        sum(final_actor_correct_counts) / final_actor_total
        if final_actor_total else 0.0
    )
    best_actor_oracle_accuracy = (
        sum(1 for value in final_actor_any_correct if value) / len(samples)
        if samples else 0.0
    )

    parsing_diagnostics = {
        "actor_answer_sources_by_round": {
            str(r): source_rates(actor_sources_by_round[r])
            for r in range(num_rounds)
        },
        "actor_final_result_rate_by_round": [
            source_rates(actor_sources_by_round[r])["final_result_rate"]
            for r in range(num_rounds)
        ],
        "actor_flexible_parse_rate_by_round": [
            source_rates(actor_sources_by_round[r])["flexible_parse_rate"]
            for r in range(num_rounds)
        ],
        "actor_unresolved_rate_by_round": [
            source_rates(actor_sources_by_round[r])["none_rate"]
            for r in range(num_rounds)
        ],
    }

    deliberation_dynamics = {
        "stayed_correct": sum(
            1 for d in details if d["initially_correct"] and d["finally_correct"]
        ),
        "flipped_to_correct": sum(1 for d in details if d["flipped_to_correct"]),
        "flipped_to_wrong": sum(1 for d in details if d["flipped_to_wrong"]),
        "stayed_wrong": sum(
            1 for d in details if not d["initially_correct"] and not d["finally_correct"]
        ),
    }

    critic_metrics = {
        "critic_confidence_parse_rate": (
            critic_confidence_parsed / critic_total if critic_total else 0.0
        ),
        "critic_suggested_answer_parse_rate": (
            critic_suggested_parsed / critic_total if critic_total else 0.0
        ),
        "critic_answer_correct_parse_rate": (
            critic_answer_correct_parsed / critic_total if critic_total else 0.0
        ),
        "critic_selected_rate": (
            critic_selected_total / critic_total if critic_total else 0.0
        ),
        "critic_total_count": critic_total,
        "critic_confidence_parsed_count": critic_confidence_parsed,
        "critic_suggested_answer_parsed_count": critic_suggested_parsed,
        "critic_answer_correct_parsed_count": critic_answer_correct_parsed,
        "critic_parse_error_counts": dict(critic_parse_errors),
    }

    return EvalResult(
        initial_accuracy=initial_acc,
        final_consensus_accuracy=final_acc,
        per_round_accuracy=round_accs,
        relative_improvement=(final_acc - initial_acc) / initial_acc if initial_acc > 0 else 0.0,
        absolute_improvement=final_acc - initial_acc,
        mean_actor_final_accuracy=mean_actor_final_accuracy,
        best_actor_oracle_accuracy=best_actor_oracle_accuracy,
        diversity_score=compute_diversity([answer or "" for answer in all_final_answers]),
        ci_95=ci_95,
        num_samples=len(samples),
        eval_time_seconds=time.time() - start_time,
        sample_details=details,
        parsing_diagnostics=parsing_diagnostics,
        deliberation_dynamics=deliberation_dynamics,
        per_round_consensus_confidence=per_round_consensus_confidence,
        critic_metrics=critic_metrics,
    )


def _build_results_payload(
    ablation_results: Dict[str, EvalResult],
    total_time: Optional[float] = None,
) -> Dict[str, Any]:
    first_result = list(ablation_results.values())[0] if ablation_results else None
    main_result = ablation_results.get("A5_full_system") or first_result
    main_metrics = {}
    parsing_diagnostics = {}
    deliberation_dynamics = {}
    critic_metrics = {}
    if main_result:
        main_metrics = {
            "initial_accuracy": main_result.initial_accuracy,
            "final_consensus_accuracy": main_result.final_consensus_accuracy,
            "absolute_improvement": main_result.absolute_improvement,
            "relative_improvement": main_result.relative_improvement,
            "ci_95": list(main_result.ci_95),
            "mean_actor_final_accuracy": main_result.mean_actor_final_accuracy,
            "best_actor_oracle_accuracy": main_result.best_actor_oracle_accuracy,
        }
        parsing_diagnostics = main_result.parsing_diagnostics
        deliberation_dynamics = main_result.deliberation_dynamics
        critic_metrics = main_result.critic_metrics
    return {
        "main_metrics": main_metrics,
        "parsing_diagnostics": parsing_diagnostics,
        "deliberation_dynamics": deliberation_dynamics,
        "critic_metrics": critic_metrics,
        "per_round_accuracy": main_result.per_round_accuracy if main_result else [],
        "per_round_consensus_confidence": (
            main_result.per_round_consensus_confidence if main_result else []
        ),
        "ablation_results": {
            name: {
                "initial_accuracy": r.initial_accuracy,
                "final_consensus_accuracy": r.final_consensus_accuracy,
                "per_round_accuracy": r.per_round_accuracy,
                "relative_improvement": r.relative_improvement,
                "absolute_improvement": r.absolute_improvement,
                "mean_actor_final_accuracy": r.mean_actor_final_accuracy,
                "best_actor_oracle_accuracy": r.best_actor_oracle_accuracy,
                "diversity_score": r.diversity_score,
                "ci_95": list(r.ci_95),
                "num_samples": r.num_samples,
                "eval_time_seconds": r.eval_time_seconds,
                "parsing_diagnostics": r.parsing_diagnostics,
                "deliberation_dynamics": r.deliberation_dynamics,
                "critic_metrics": r.critic_metrics,
                "ablation_metadata": r.ablation_metadata,
            }
            for name, r in ablation_results.items()
        },
        "total_eval_time_seconds": total_time,
        "sample_details_by_ablation": {
            name: r.sample_details for name, r in ablation_results.items()
        },
    }


def _save_results(
    results_file: str,
    ablation_results: Dict[str, EvalResult],
    total_time: Optional[float] = None,
):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(_build_results_payload(ablation_results, total_time), f, indent=2)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _mean_list(lists: List[List[float]]) -> List[float]:
    if not lists:
        return []
    width = max(len(values) for values in lists)
    result = []
    for i in range(width):
        values = [items[i] for items in lists if i < len(items)]
        result.append(_mean(values))
    return result


def _sum_counter_dicts(dicts: List[Dict[str, int]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in dicts:
        counter.update(item or {})
    return dict(counter)


def _aggregate_source_metric_dicts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}

    aggregate: Dict[str, Any] = {}
    total = sum(int(item.get("total", 0)) for item in items)
    aggregate["total"] = total
    source_keys = (
        "final_result",
        "final_answer",
        "boxed",
        "tail_claim",
        "weak_tail",
        "none",
        "flexible_parse",
    )
    for key in source_keys:
        count_key = f"{key}_count"
        rate_key = f"{key}_rate"
        count = sum(int(item.get(count_key, 0)) for item in items)
        aggregate[count_key] = count
        aggregate[rate_key] = count / total if total else 0.0
    parse_success_count = sum(int(item.get("parse_success_count", 0)) for item in items)
    aggregate["parse_success_count"] = parse_success_count
    aggregate["parse_success_rate"] = parse_success_count / total if total else 0.0
    return aggregate


def _aggregate_parsing_diagnostics(results: List[EvalResult]) -> Dict[str, Any]:
    if not results:
        return {}

    round_keys = sorted({
        int(key)
        for result in results
        for key in result.parsing_diagnostics.get("actor_answer_sources_by_round", {})
    })
    sources_by_round = {}
    for round_key in round_keys:
        sources_by_round[str(round_key)] = _aggregate_source_metric_dicts([
            result.parsing_diagnostics.get("actor_answer_sources_by_round", {}).get(
                str(round_key),
                {},
            )
            for result in results
        ])

    return {
        "actor_answer_sources_by_round": sources_by_round,
        "actor_final_result_rate_by_round": [
            sources_by_round[str(r)].get("final_result_rate", 0.0)
            for r in round_keys
        ],
        "actor_flexible_parse_rate_by_round": [
            sources_by_round[str(r)].get("flexible_parse_rate", 0.0)
            for r in round_keys
        ],
        "actor_unresolved_rate_by_round": [
            sources_by_round[str(r)].get("none_rate", 0.0)
            for r in round_keys
        ],
    }


def _aggregate_critic_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    critic_total = sum(
        int(result.critic_metrics.get("critic_total_count", 0))
        for result in results
    )
    confidence_parsed = sum(
        int(result.critic_metrics.get("critic_confidence_parsed_count", 0))
        for result in results
    )
    suggested_parsed = sum(
        int(result.critic_metrics.get("critic_suggested_answer_parsed_count", 0))
        for result in results
    )
    answer_correct_parsed = sum(
        int(result.critic_metrics.get("critic_answer_correct_parsed_count", 0))
        for result in results
    )
    selected_rate_sum = sum(
        float(result.critic_metrics.get("critic_selected_rate", 0.0))
        * int(result.critic_metrics.get("critic_total_count", 0))
        for result in results
    )
    return {
        "critic_confidence_parse_rate": (
            confidence_parsed / critic_total if critic_total else 0.0
        ),
        "critic_suggested_answer_parse_rate": (
            suggested_parsed / critic_total if critic_total else 0.0
        ),
        "critic_answer_correct_parse_rate": (
            answer_correct_parsed / critic_total if critic_total else 0.0
        ),
        "critic_selected_rate": (
            selected_rate_sum / critic_total if critic_total else 0.0
        ),
        "critic_total_count": critic_total,
        "critic_confidence_parsed_count": confidence_parsed,
        "critic_suggested_answer_parsed_count": suggested_parsed,
        "critic_answer_correct_parsed_count": answer_correct_parsed,
        "critic_parse_error_counts": _sum_counter_dicts([
            result.critic_metrics.get("critic_parse_error_counts", {})
            for result in results
        ]),
    }


def _aggregate_eval_results(
    name: str,
    component_results: List[tuple[Dict[str, Any], EvalResult]],
) -> EvalResult:
    """Aggregate repeated ablation runs into one mean/std result."""
    results = [result for _, result in component_results]
    if not results:
        raise ValueError(f"Cannot aggregate empty ablation results for {name}")

    final_values = [r.final_consensus_accuracy for r in results]
    initial_values = [r.initial_accuracy for r in results]
    relative_values = [r.relative_improvement for r in results]
    absolute_values = [r.absolute_improvement for r in results]
    mean_actor_values = [r.mean_actor_final_accuracy for r in results]
    best_actor_values = [r.best_actor_oracle_accuracy for r in results]
    diversity_values = [r.diversity_score for r in results]

    final_mean = _mean(final_values)
    final_std = _std(final_values)
    initial_mean = _mean(initial_values)
    relative_mean = _mean(relative_values)
    absolute_mean = _mean(absolute_values)

    component_summaries = []
    for metadata, result in component_results:
        component_summaries.append({
            **metadata,
            "initial_accuracy": result.initial_accuracy,
            "final_consensus_accuracy": result.final_consensus_accuracy,
            "relative_improvement": result.relative_improvement,
            "absolute_improvement": result.absolute_improvement,
            "mean_actor_final_accuracy": result.mean_actor_final_accuracy,
            "best_actor_oracle_accuracy": result.best_actor_oracle_accuracy,
            "ci_95": list(result.ci_95),
            "eval_time_seconds": result.eval_time_seconds,
        })

    return EvalResult(
        initial_accuracy=initial_mean,
        final_consensus_accuracy=final_mean,
        per_round_accuracy=_mean_list([r.per_round_accuracy for r in results]),
        relative_improvement=relative_mean,
        absolute_improvement=absolute_mean,
        mean_actor_final_accuracy=_mean(mean_actor_values),
        best_actor_oracle_accuracy=_mean(best_actor_values),
        diversity_score=_mean(diversity_values),
        ci_95=(
            max(0.0, final_mean - 1.96 * final_std),
            min(1.0, final_mean + 1.96 * final_std),
        ),
        num_samples=results[0].num_samples,
        eval_time_seconds=sum(r.eval_time_seconds for r in results),
        sample_details=[],
        parsing_diagnostics=_aggregate_parsing_diagnostics(results),
        deliberation_dynamics={
            key: sum(result.deliberation_dynamics.get(key, 0) for result in results)
            for key in (
                "stayed_correct",
                "flipped_to_correct",
                "flipped_to_wrong",
                "stayed_wrong",
            )
        },
        per_round_consensus_confidence=_mean_list([
            r.per_round_consensus_confidence for r in results
        ]),
        critic_metrics=_aggregate_critic_metrics(results),
        ablation_metadata={
            "name": name,
            "aggregation": "mean_std_over_component_runs",
            "num_component_runs": len(results),
            "component_results": component_summaries,
            "std": {
                "initial_accuracy": _std(initial_values),
                "final_consensus_accuracy": final_std,
                "relative_improvement": _std(relative_values),
                "absolute_improvement": _std(absolute_values),
                "mean_actor_final_accuracy": _std(mean_actor_values),
                "best_actor_oracle_accuracy": _std(best_actor_values),
                "diversity_score": _std(diversity_values),
            },
        },
    )


# ============================================================
# Main evaluation with shared engine
# ============================================================

def run_all_evaluations(
    registry: Dict[str, Any],
    samples: List[Dict],
    dataset_name: str,
    num_rounds: int,
    max_tokens: int,
    temperature: float,
    devices: List[int],
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    run_ablations: bool,
    max_model_len: int = 4096,
    max_lora_rank: int = 256,
    router_top_k: int = 2,
    router_min_confidence: float = 0.1,
    router_fallback_to_uniform: bool = False,
    route_feedback_to_actor: bool = True,
    consensus_uses_selected_critics_only: bool = False,
    phase_actor_dir: str = "output/society/actors",
    phase_critic_dir: str = "output/society/critics",
    results_file: Optional[str] = None,
    eval_batch_size: int = 1,
) -> Dict[str, EvalResult]:
    """Load base model ONCE, run all evaluations sharing the same engine.

    Ablation design (clean causal isolation):
      A0: Base model only (no LoRA, no training) — zero-training reference
      A1: 1 Actor + 1 Critic from PHASE 3/4 diversification (pre-society-training)
          → Isolates basic diversification effect vs base model
      A2: diverse Actors (phase 3) + 1 Critic (phase 4) — Actor diversity only
          → Isolates Actor diversity contribution vs A1
      A3: 1 Actor (phase 3) + 4 Critics (phase 4) + Router — Critic specialization only
          → Isolates Critic specialization contribution vs A1
      A4: Actors + 4 Critics from FINAL registry, uniform weights — full agents, no routing
          → Shows society training + diversity effect without Router
      A5: Actors + 4 Critics from FINAL registry + Router — complete system
          → Full system with all components

    Key: A1-A3 use pre-society-training LoRA (phase 3/4 registries),
         A4-A5 use post-society-training LoRA (final joint registry).
    """
    from src.inference.vllm_server import VLLMInference

    base_model = registry.get("training_config", {}).get("base_model", "Qwen/Qwen3-14B")
    all_actor_names = list(registry.get("actors", {}).keys())
    all_critic_names = list(registry.get("critics", {}).keys())

    results: Dict[str, EvalResult] = {}

    # Local helper to avoid passing eval_batch_size to every call site
    def _run(engine, ac, cc, samples, dn, lora, nr, mt, temp, **kw):
        return _run_deliberation_on_samples(
            engine, ac, cc, samples, dn, lora, nr, mt, temp,
            eval_batch_size=eval_batch_size,
            **kw,
        )

    # Load phase registries for A1-A3 ablations (pre-society-training LoRA)
    phase_actors, phase_critics, phase_lora = _build_agent_configs_from_phase_registries(
        actor_phase_dir=phase_actor_dir,
        critic_phase_dir=phase_critic_dir,
        base_model=base_model,
    )
    final_lora_paths = [
        info.get("model_path", "")
        for info in list(registry.get("actors", {}).values())
        + list(registry.get("critics", {}).values())
        if info.get("model_path", "")
    ]
    all_lora_paths = set(final_lora_paths) | set(phase_lora.values())
    total_agents_with_lora = max(len(all_lora_paths), 1)

    logger.info(f"Loading base model ONCE: {base_model}")
    logger.info(
        f"  LoRA adapters: {total_agents_with_lora} unique paths "
        f"(final={len(final_lora_paths)}, phase={len(phase_lora)})"
    )
    engine = VLLMInference(
        base_model,
        cuda_device=devices,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=True,
        max_loras=total_agents_with_lora,
        max_lora_rank=max_lora_rank,
    )

    logger.info(
        f"  Phase registries: {len(phase_actors)} actors, {len(phase_critics)} critics "
        f"(for A1-A3 ablations)"
    )

    try:
        if run_ablations:
            # A0: Base model baseline (no training at all) — zero-training reference
            logger.info("[A0] Base model baseline (no LoRA, no training)...")
            a_configs, c_configs, lora = _build_base_agent_configs(base_model)
            results["A0_base_model"] = _run(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=1,
                router_min_confidence=router_min_confidence,
                router_fallback_to_uniform=router_fallback_to_uniform,
                route_feedback_to_actor=route_feedback_to_actor,
                consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
            )
            logger.info(
                "  A0: initial="
                f"{results['A0_base_model'].initial_accuracy:.3f} "
                "final_consensus="
                f"{results['A0_base_model'].final_consensus_accuracy:.3f}"
            )
            if results_file:
                _save_results(results_file, results)

            # A1: 1 Actor + 1 Critic from phase registries (pre-society-training)
            # This is the true ACC-Collab baseline: independently diversified
            # single agent pair, NOT a subset of the jointly trained system.
            logger.info(
                "[A1] All phase Actor x phase Critic single-pair runs "
                "(ACC-Collab baseline)..."
            )
            if phase_actors and phase_critics:
                component_results = []
                for actor in phase_actors:
                    for critic in phase_critics:
                        logger.info(f"  A1 component: actor={actor.name}, critic={critic.name}")
                        component_lora = {
                            name: path
                            for name, path in phase_lora.items()
                            if name in {actor.name, critic.name}
                        }
                        component_result = _run(
                            engine,
                            [actor],
                            [critic],
                            samples, dataset_name,
                            component_lora, num_rounds, max_tokens, temperature,
                            router_top_k=1,
                            router_min_confidence=router_min_confidence,
                            router_fallback_to_uniform=router_fallback_to_uniform,
                            route_feedback_to_actor=route_feedback_to_actor,
                            consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
                        )
                        component_results.append((
                            {"actor": actor.name, "critic": critic.name},
                            component_result,
                        ))
                results["A1_acc_collab"] = _aggregate_eval_results(
                    "A1_acc_collab",
                    component_results,
                )
            else:
                logger.warning("  Phase registries empty, falling back to final registry single-pair sweep")
                a_configs, c_configs, lora = _build_agent_configs(
                    registry,
                )
                component_results = []
                for actor in a_configs:
                    for critic in c_configs:
                        component_lora = {
                            name: path
                            for name, path in lora.items()
                            if name in {actor.name, critic.name}
                        }
                        component_result = _run(
                            engine, [actor], [critic], samples, dataset_name,
                            component_lora, num_rounds, max_tokens, temperature,
                            router_top_k=1,
                            router_min_confidence=router_min_confidence,
                            router_fallback_to_uniform=router_fallback_to_uniform,
                            route_feedback_to_actor=route_feedback_to_actor,
                            consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
                        )
                        component_results.append((
                            {"actor": actor.name, "critic": critic.name},
                            component_result,
                        ))
                results["A1_acc_collab"] = _aggregate_eval_results(
                    "A1_acc_collab",
                    component_results,
                )
            logger.info(
                "  A1: initial="
                f"{results['A1_acc_collab'].initial_accuracy:.3f} "
                "final_consensus="
                f"{results['A1_acc_collab'].final_consensus_accuracy:.3f} "
                "+/- "
                f"{results['A1_acc_collab'].ablation_metadata['std']['final_consensus_accuracy']:.3f}"
            )
            if results_file:
                _save_results(results_file, results)

            # A2: diverse Actors (phase 3) + 1 Critic (phase 4) — Actor diversity only
            # Uses PRE-society-training LoRA from phase 3/4 diversification.
            # Causal question: do diverse Actors improve over 1 Actor?
            logger.info(
                "[A2] Phase-diversified Actors + each fixed phase Critic "
                "(Actor diversity)..."
            )
            if phase_actors and phase_critics:
                component_results = []
                actor_names = {actor.name for actor in phase_actors}
                for critic in phase_critics:
                    logger.info(f"  A2 component: critic={critic.name}")
                    component_lora = {
                        name: path
                        for name, path in phase_lora.items()
                        if name in actor_names or name == critic.name
                    }
                    component_result = _run(
                        engine,
                        phase_actors,
                        [critic],
                        samples, dataset_name,
                        component_lora, num_rounds, max_tokens, temperature,
                        router_top_k=1,
                        router_min_confidence=router_min_confidence,
                        router_fallback_to_uniform=router_fallback_to_uniform,
                        route_feedback_to_actor=route_feedback_to_actor,
                        consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
                    )
                    component_results.append((
                        {
                            "actors": [actor.name for actor in phase_actors],
                            "critic": critic.name,
                        },
                        component_result,
                    ))
                results["A2_actor_diversity"] = _aggregate_eval_results(
                    "A2_actor_diversity",
                    component_results,
                )
            else:
                logger.warning("  Phase registries empty, falling back to final registry subset")
                a_configs, _, a_lora = _build_agent_configs(
                    registry, actor_names=all_actor_names,
                )
                _, c_configs, c_lora = _build_agent_configs(registry, critic_names=all_critic_names)
                component_results = []
                for critic in c_configs:
                    component_lora = {
                        **a_lora,
                        **{critic.name: c_lora.get(critic.name, "")},
                    }
                    component_lora = {
                        name: path for name, path in component_lora.items() if path
                    }
                    component_result = _run(
                        engine, a_configs, [critic], samples, dataset_name,
                        component_lora, num_rounds, max_tokens, temperature,
                        router_top_k=1,
                        router_min_confidence=router_min_confidence,
                        router_fallback_to_uniform=router_fallback_to_uniform,
                        route_feedback_to_actor=route_feedback_to_actor,
                        consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
                    )
                    component_results.append((
                        {
                            "actors": [actor.name for actor in a_configs],
                            "critic": critic.name,
                        },
                        component_result,
                    ))
                results["A2_actor_diversity"] = _aggregate_eval_results(
                    "A2_actor_diversity",
                    component_results,
                )
            logger.info(
                "  A2: initial="
                f"{results['A2_actor_diversity'].initial_accuracy:.3f} "
                "final_consensus="
                f"{results['A2_actor_diversity'].final_consensus_accuracy:.3f} "
                "+/- "
                f"{results['A2_actor_diversity'].ablation_metadata['std']['final_consensus_accuracy']:.3f}"
            )
            if results_file:
                _save_results(results_file, results)

            # A3: 1 Actor (phase 3) + 4 Critics (phase 4) + Router — Critic specialization only
            # Uses PRE-society-training LoRA from phase 3/4 diversification.
            # Causal question: does Critic specialization + Router improve over 1 Critic?
            logger.info(
                "[A3] Each fixed phase Actor + phase-diversified Critics + Router "
                "(Critic specialization)..."
            )
            if phase_actors and phase_critics:
                component_results = []
                critic_names = {critic.name for critic in phase_critics}
                for actor in phase_actors:
                    logger.info(f"  A3 component: actor={actor.name}")
                    component_lora = {
                        name: path
                        for name, path in phase_lora.items()
                        if name == actor.name or name in critic_names
                    }
                    component_result = _run(
                        engine,
                        [actor],
                        phase_critics,
                        samples, dataset_name,
                        component_lora, num_rounds, max_tokens, temperature,
                        router_top_k=min(2, len(phase_critics)),
                        router_min_confidence=router_min_confidence,
                        router_fallback_to_uniform=router_fallback_to_uniform,
                        route_feedback_to_actor=route_feedback_to_actor,
                        consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
                    )
                    component_results.append((
                        {
                            "actor": actor.name,
                            "critics": [critic.name for critic in phase_critics],
                        },
                        component_result,
                    ))
                results["A3_critic_specialization"] = _aggregate_eval_results(
                    "A3_critic_specialization",
                    component_results,
                )
            else:
                logger.warning("  Phase registries empty, falling back to final registry subset")
                a_configs, _, a_lora = _build_agent_configs(registry, actor_names=all_actor_names)
                _, c_configs, c_lora = _build_agent_configs(registry, critic_names=all_critic_names)
                component_results = []
                for actor in a_configs:
                    component_lora = {
                        **{actor.name: a_lora.get(actor.name, "")},
                        **c_lora,
                    }
                    component_lora = {
                        name: path for name, path in component_lora.items() if path
                    }
                    component_result = _run(
                        engine, [actor], c_configs, samples, dataset_name,
                        component_lora, num_rounds, max_tokens, temperature,
                        router_top_k=min(2, len(c_configs)),
                        router_min_confidence=router_min_confidence,
                        router_fallback_to_uniform=router_fallback_to_uniform,
                        route_feedback_to_actor=route_feedback_to_actor,
                        consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
                    )
                    component_results.append((
                        {
                            "actor": actor.name,
                            "critics": [critic.name for critic in c_configs],
                        },
                        component_result,
                    ))
                results["A3_critic_specialization"] = _aggregate_eval_results(
                    "A3_critic_specialization",
                    component_results,
                )
            logger.info(
                "  A3: initial="
                f"{results['A3_critic_specialization'].initial_accuracy:.3f} "
                "final_consensus="
                f"{results['A3_critic_specialization'].final_consensus_accuracy:.3f} "
                "+/- "
                f"{results['A3_critic_specialization'].ablation_metadata['std']['final_consensus_accuracy']:.3f}"
            )
            if results_file:
                _save_results(results_file, results)

            # A4: Actors + 4 Critics from FINAL registry, uniform weights (no routing)
            # Uses POST-society-training LoRA — shows joint training + diversity effect.
            # Key difference from A5: uniform_weights=True means all critics
            # contribute equally (no softmax confidence gating)
            logger.info("[A4] Actors + 4 Critics from final registry (no routing, uniform weights)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names,
                critic_names=all_critic_names,
            )
            results["A4_no_routing"] = _run(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=len(c_configs),  # Use ALL critics
                router_uniform=True,  # Equal weights (no softmax)
                router_min_confidence=router_min_confidence,
                router_fallback_to_uniform=router_fallback_to_uniform,
                route_feedback_to_actor=route_feedback_to_actor,
                consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
            )
            logger.info(
                "  A4: initial="
                f"{results['A4_no_routing'].initial_accuracy:.3f} "
                "final_consensus="
                f"{results['A4_no_routing'].final_consensus_accuracy:.3f}"
            )
            if results_file:
                _save_results(results_file, results)

            # A5: Actors + 4 Critics from FINAL registry + Router (full system)
            logger.info("[A5] Actors + 4 Critics + Router (full system)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names,
                critic_names=all_critic_names,
            )
            results["A5_full_system"] = _run(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=router_top_k,  # From config
                router_uniform=False,       # Softmax confidence weighting
                router_min_confidence=router_min_confidence,
                router_fallback_to_uniform=router_fallback_to_uniform,
                route_feedback_to_actor=route_feedback_to_actor,
                consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
            )
            logger.info(
                "  A5: initial="
                f"{results['A5_full_system'].initial_accuracy:.3f} "
                "final_consensus="
                f"{results['A5_full_system'].final_consensus_accuracy:.3f}"
            )
            if results_file:
                _save_results(results_file, results)
        else:
            # Single main evaluation with full system
            logger.info("[Main] Full system evaluation...")
            a_configs, c_configs, lora = _build_agent_configs(registry)
            results["main"] = _run(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=router_top_k,
                router_min_confidence=router_min_confidence,
                router_fallback_to_uniform=router_fallback_to_uniform,
                route_feedback_to_actor=route_feedback_to_actor,
                consensus_uses_selected_critics_only=consensus_uses_selected_critics_only,
            )

    finally:
        del engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def print_qualitative_examples(result: EvalResult, num_examples: int):
    """Print qualitative examples for analysis."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Qualitative Examples")
    logger.info("=" * 60)

    details = result.sample_details[:num_examples]

    for i, detail in enumerate(details, 1):
        logger.info(f"\n[Example {i}]")
        logger.info(f"  Question: {detail['question'][:100]}...")
        logger.info(f"  Predicted: {detail['final_answer']}")
        logger.info(f"  Ground Truth: {detail['ground_truth']}")
        logger.info(f"  Confidence: {detail.get('confidence', 0.0):.2f}")


def main():
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Society Evaluation")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Society dir: {args.society_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Run ablations: {args.run_ablations}")
    logger.info("=" * 60)

    runtime = resolve_evaluation_runtime(args)
    logger.info(
        "  Evaluation mode: "
        f"{runtime.mode} "
        f"(devices={runtime.devices}, "
        f"tensor_parallel_size={runtime.tensor_parallel_size}, "
        f"gpu_memory_utilization={runtime.gpu_memory_utilization})"
    )

    # Load society registry
    logger.info("[Step 1] Loading society registry...")
    registry = load_society_registry(args.society_dir)

    # Load dataset
    logger.info("[Step 2] Loading dataset...")
    samples = load_eval_dataset(
        args.dataset, args.seed,
        sampling=getattr(args, "sampling", None),
        mmlu_load_mode=getattr(args, "mmlu_load_mode", "by_subject"),
    )
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive or None, got {max_samples}")
        if len(samples) > max_samples:
            samples = samples[:max_samples]
            logger.info(f"  Applied max_samples={max_samples}")
    logger.info(f"  Test samples: {len(samples)}")

    # Run all evaluations with shared engine
    logger.info("[Step 3] Running evaluation (shared base model)...")
    total_start = time.time()
    results_file = os.path.join(output_dir, "results.json")

    # Resolve phase registry directories for pre-society-training ablations
    cache_dir = getattr(args, "cache_dir", "output/society")
    phase_actor_dir = os.path.join(cache_dir, "actors")
    phase_critic_dir = os.path.join(cache_dir, "critics")

    ablation_results = run_all_evaluations(
        registry=registry,
        samples=samples,
        dataset_name=args.dataset,
        num_rounds=args.num_rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        devices=runtime.devices,
        tensor_parallel_size=runtime.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=runtime.gpu_memory_utilization,
        run_ablations=args.run_ablations,
        max_model_len=args.max_model_len,
        max_lora_rank=args.max_lora_rank,
        router_top_k=args.router_top_k,
        router_min_confidence=getattr(args, "router_min_confidence", 0.1),
        router_fallback_to_uniform=getattr(args, "router_fallback_to_uniform", False),
        route_feedback_to_actor=getattr(args, "route_feedback_to_actor", True),
        consensus_uses_selected_critics_only=getattr(
            args,
            "consensus_uses_selected_critics_only",
            False,
        ),
        phase_actor_dir=phase_actor_dir,
        phase_critic_dir=phase_critic_dir,
        results_file=results_file,
        eval_batch_size=getattr(args, "eval_batch_size", 1),
    )
    total_time = time.time() - total_start
    logger.info(f"Total evaluation time: {total_time:.1f}s")

    # Save results
    logger.info("[Step 4] Saving results...")

    first_result = list(ablation_results.values())[0] if ablation_results else None
    _save_results(results_file, ablation_results, total_time)

    logger.info(f"  Results saved: {results_file}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    for name, result in ablation_results.items():
        logger.info(f"\n[{name}]")
        logger.info(f"  Initial Accuracy:  {result.initial_accuracy:.4f}")
        logger.info(f"  Final Consensus Accuracy: {result.final_consensus_accuracy:.4f}")
        logger.info(f"  Relative Improvement:     {result.relative_improvement:.4f}")
        logger.info(f"  Absolute Gain:     {result.absolute_improvement:+.4f}")
        logger.info(f"  95% CI:            ({result.ci_95[0]:.4f}, {result.ci_95[1]:.4f})")
        logger.info(f"  Diversity Score:   {result.diversity_score:.4f}")
        logger.info(f"  Eval Time:         {result.eval_time_seconds:.1f}s")

    # Print qualitative examples
    main_result = ablation_results.get("A5_full_system") or first_result
    if main_result:
        print_qualitative_examples(main_result, args.num_samples_for_qualitative)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
