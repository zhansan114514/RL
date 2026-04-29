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
        --config configs/society/experiment_h100.yaml \
        --run_ablations
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "society_dir": "output/society/society",
    "output_dir": "output/society/eval",
    "num_rounds": 2,
    "max_tokens": 512,
    "temperature": 0.0,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
    "run_ablations": True,
    "num_samples_for_qualitative": 5,
    "max_model_len": 4096,
    "max_lora_rank": 256,
    "router_top_k": 2,
}


@dataclass
class EvalResult:
    """Result of society evaluation."""
    initial_accuracy: float
    final_accuracy: float
    per_round_accuracy: List[float]
    improvement_rate: float
    absolute_improvement: float
    consensus_accuracy: float
    diversity_score: float
    ci_95: tuple[float, float]
    num_samples: int
    eval_time_seconds: float
    sample_details: List[Dict[str, Any]]


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Society evaluation",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--no_ablations", action="store_true",
        help="Skip ablation experiments.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step06_evaluate", defaults=STEP_DEFAULTS).to_namespace()

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


def load_dataset(dataset_name: str, seed: int, max_samples: Optional[int]) -> List[Dict]:
    """Load test dataset."""
    from src.data.loader import load_dataset

    data = load_dataset(dataset_name, seed=seed)
    test_data = data.get("test", []) or data.get("validation", [])

    if max_samples:
        test_data = test_data[:max_samples]

    return test_data


def compute_diversity(responses: List[str]) -> float:
    """Compute diversity score using unique response ratio."""
    if not responses:
        return 0.0
    return len(set(responses)) / len(responses)


def _compute_ci(
    predictions: List[str],
    labels: List[str],
    task_type: str,
) -> tuple:
    """Compute accuracy and Wilson confidence interval margin."""
    from src.algorithms.reward import math_answers_equal

    correct = 0
    for pred, label in zip(predictions, labels):
        if task_type == "math":
            if math_answers_equal(pred or "", label):
                correct += 1
        else:
            if (pred or "").upper() == label.upper():
                correct += 1

    n = len(labels)
    accuracy = correct / n if n > 0 else 0.0

    z = 1.96
    if n == 0:
        return accuracy, 0.0
    denom = 1 + z**2 / n
    margin = z * math.sqrt((accuracy * (1 - accuracy) + z**2 / (4 * n)) / n) / denom
    return accuracy, margin


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
    base_model = registry.get("training_config", {}).get("base_model", "Qwen/Qwen2.5-7B-Instruct")

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
) -> EvalResult:
    """Run deliberation on samples with a shared vLLM engine. No model loading."""
    from src.society.multi_deliberation import multi_agent_deliberate_single_gpu
    from src.society.router import CriticRouter

    # Create router with the specified configuration
    router = CriticRouter(top_k=router_top_k, uniform_weights=router_uniform)

    start_time = time.time()
    all_final_answers = []
    all_initial_answers = []
    all_responses = []
    details = []
    # Track per-round consensus answers for per-round accuracy
    per_round_answers: Dict[int, List[str]] = {r: [] for r in range(num_rounds)}

    for si, sample in enumerate(samples):
        if (si + 1) % 5 == 0 or si == 0:
            logger.info(f"    Sample {si + 1}/{len(samples)}")

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
        )

        final_answer = result.consensus_answer or ""
        all_final_answers.append(final_answer)
        all_responses.append(final_answer)

        # Collect per-round consensus answers
        for rnd in result.rounds:
            if rnd.round_num in per_round_answers:
                per_round_answers[rnd.round_num].append(
                    rnd.consensus_answer or final_answer
                )

        # Compute initial answer as majority vote across all actors (not just first)
        initial_answer = final_answer
        if result.rounds:
            first_round = result.rounds[0]
            init_answers = [a for a in first_round.actor_answers.values() if a is not None]
            if init_answers:
                counter = Counter(init_answers)
                initial_answer = counter.most_common(1)[0][0]
            else:
                initial_answer = final_answer
        all_initial_answers.append(initial_answer)

        details.append({
            "question": sample.get("question", ""),
            "final_answer": final_answer,
            "confidence": result.consensus_confidence,
            "ground_truth": sample.get("answer", ""),
        })

    labels = [s.get("answer", "") for s in samples]
    task_type = samples[0].get("task_type", "yes_no") if samples else "yes_no"

    initial_acc, _ = _compute_ci(all_initial_answers, labels, task_type)
    final_acc, ci_margin = _compute_ci(all_final_answers, labels, task_type)
    ci_95 = (max(0, final_acc - ci_margin), min(1, final_acc + ci_margin))

    # Compute true per-round accuracy from consensus answers
    round_accs = []
    for r in range(num_rounds):
        if per_round_answers[r]:
            acc, _ = _compute_ci(per_round_answers[r], labels[:len(per_round_answers[r])], task_type)
            round_accs.append(acc)
    if not round_accs:
        round_accs = [initial_acc, final_acc]

    return EvalResult(
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        per_round_accuracy=round_accs,
        improvement_rate=(final_acc - initial_acc) / initial_acc if initial_acc > 0 else 0.0,
        absolute_improvement=final_acc - initial_acc,
        consensus_accuracy=final_acc,
        diversity_score=compute_diversity(all_responses),
        ci_95=ci_95,
        num_samples=len(samples),
        eval_time_seconds=time.time() - start_time,
        sample_details=details,
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
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    run_ablations: bool,
    max_model_len: int = 4096,
    max_lora_rank: int = 256,
    router_top_k: int = 2,
    phase_actor_dir: str = "output/society/actors",
    phase_critic_dir: str = "output/society/critics",
) -> Dict[str, EvalResult]:
    """Load base model ONCE, run all evaluations sharing the same engine.

    Ablation design (clean causal isolation):
      A0: Base model only (no LoRA, no training) — zero-training reference
      A1: 1 Actor + 1 Critic from PHASE 3/4 diversification (pre-society-training)
          → Isolates basic diversification effect vs base model
      A2: 3 diverse Actors (phase 3) + 1 Critic (phase 4) — Actor diversity only
          → Isolates Actor diversity contribution vs A1
      A3: 1 Actor (phase 3) + 4 Critics (phase 4) + Router — Critic specialization only
          → Isolates Critic specialization contribution vs A1
      A4: 3 Actors + 4 Critics from FINAL registry, uniform weights — full agents, no routing
          → Shows society training + diversity effect without Router
      A5: 3 Actors + 4 Critics from FINAL registry + Router — complete system
          → Full system with all components

    Key: A1-A3 use pre-society-training LoRA (phase 3/4 registries),
         A4-A5 use post-society-training LoRA (final joint registry).
    """
    from src.inference.vllm_server import VLLMInference

    base_model = registry.get("training_config", {}).get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    all_actor_names = list(registry.get("actors", {}).keys())
    all_critic_names = list(registry.get("critics", {}).keys())

    results: Dict[str, EvalResult] = {}

    # Count total agents that will need LoRA across all ablation configs
    # (A2-A5 use LoRA-enabled agents, A1 uses base model only)
    total_agents_with_lora = len(all_actor_names) + len(all_critic_names)

    logger.info(f"Loading base model ONCE: {base_model}")
    logger.info(f"  LoRA agents: {total_agents_with_lora} ({len(all_actor_names)} actors + {len(all_critic_names)} critics)")
    engine = VLLMInference(
        base_model,
        cuda_device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=True,
        max_loras=total_agents_with_lora,
        max_lora_rank=max_lora_rank,
    )

    # Load phase registries for A1-A3 ablations (pre-society-training LoRA)
    phase_actors, phase_critics, phase_lora = _build_agent_configs_from_phase_registries(
        actor_phase_dir=phase_actor_dir,
        critic_phase_dir=phase_critic_dir,
        base_model=base_model,
    )
    phase_actor_names = [a.name for a in phase_actors]
    phase_critic_names = [c.name for c in phase_critics]
    logger.info(
        f"  Phase registries: {len(phase_actors)} actors, {len(phase_critics)} critics "
        f"(for A1-A3 ablations)"
    )

    try:
        if run_ablations:
            # A0: Base model baseline (no training at all) — zero-training reference
            logger.info("[A0] Base model baseline (no LoRA, no training)...")
            a_configs, c_configs, lora = _build_base_agent_configs(base_model)
            results["A0_base_model"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=1,
            )
            logger.info(f"  A0: initial={results['A0_base_model'].initial_accuracy:.3f} final={results['A0_base_model'].final_accuracy:.3f}")

            # A1: 1 Actor + 1 Critic from phase registries (pre-society-training)
            # This is the true ACC-Collab baseline: independently diversified
            # single agent pair, NOT a subset of the jointly trained system.
            logger.info("[A1] 1 phase-diversified Actor + 1 phase-diversified Critic (ACC-Collab baseline)...")
            if phase_actors and phase_critics:
                results["A1_acc_collab"] = _run_deliberation_on_samples(
                    engine,
                    [phase_actors[0]],
                    [phase_critics[0]],
                    samples, dataset_name,
                    phase_lora, num_rounds, max_tokens, temperature,
                    router_top_k=1,
                )
            else:
                logger.warning("  Phase registries empty, falling back to first agent from final registry")
                a_configs, c_configs, lora = _build_agent_configs(
                    registry,
                    actor_names=[all_actor_names[0]],
                    critic_names=[all_critic_names[0]],
                )
                results["A1_acc_collab"] = _run_deliberation_on_samples(
                    engine, a_configs, c_configs, samples, dataset_name,
                    lora, num_rounds, max_tokens, temperature,
                    router_top_k=1,
                )
            logger.info(f"  A1: initial={results['A1_acc_collab'].initial_accuracy:.3f} final={results['A1_acc_collab'].final_accuracy:.3f}")

            # A2: 3 diverse Actors (phase 3) + 1 Critic (phase 4) — Actor diversity only
            # Uses PRE-society-training LoRA from phase 3/4 diversification.
            # Causal question: does having 3 diverse Actors improve over 1 Actor?
            logger.info("[A2] 3 phase-diversified Actors + 1 phase-diversified Critic (Actor diversity)...")
            if phase_actors and phase_critics:
                a2_lora = dict(phase_lora)
                # Keep only the first critic's lora
                for cn in phase_critic_names[1:]:
                    a2_lora.pop(cn, None)
                results["A2_actor_diversity"] = _run_deliberation_on_samples(
                    engine,
                    phase_actors,
                    [phase_critics[0]],
                    samples, dataset_name,
                    a2_lora, num_rounds, max_tokens, temperature,
                    router_top_k=1,
                )
            else:
                logger.warning("  Phase registries empty, falling back to final registry subset")
                a_configs, _, a_lora = _build_agent_configs(
                    registry, actor_names=all_actor_names[:3],
                )
                _, c_configs, c_lora = _build_agent_configs(
                    registry, critic_names=[all_critic_names[0]],
                )
                results["A2_actor_diversity"] = _run_deliberation_on_samples(
                    engine, a_configs, c_configs, samples, dataset_name,
                    {**a_lora, **c_lora}, num_rounds, max_tokens, temperature,
                    router_top_k=1,
                )
            logger.info(f"  A2: initial={results['A2_actor_diversity'].initial_accuracy:.3f} final={results['A2_actor_diversity'].final_accuracy:.3f}")

            # A3: 1 Actor (phase 3) + 4 Critics (phase 4) + Router — Critic specialization only
            # Uses PRE-society-training LoRA from phase 3/4 diversification.
            # Causal question: does Critic specialization + Router improve over 1 Critic?
            logger.info("[A3] 1 phase-diversified Actor + 4 phase-diversified Critics + Router (Critic specialization)...")
            if phase_actors and phase_critics:
                a3_lora = dict(phase_lora)
                # Keep only the first actor's lora
                for an in phase_actor_names[1:]:
                    a3_lora.pop(an, None)
                results["A3_critic_specialization"] = _run_deliberation_on_samples(
                    engine,
                    [phase_actors[0]],
                    phase_critics,
                    samples, dataset_name,
                    a3_lora, num_rounds, max_tokens, temperature,
                    router_top_k=2,
                )
            else:
                logger.warning("  Phase registries empty, falling back to final registry subset")
                a_configs, _, a_lora = _build_agent_configs(
                    registry, actor_names=[all_actor_names[0]],
                )
                _, c_configs, c_lora = _build_agent_configs(
                    registry, critic_names=all_critic_names,
                )
                results["A3_critic_specialization"] = _run_deliberation_on_samples(
                    engine, a_configs, c_configs, samples, dataset_name,
                    {**a_lora, **c_lora}, num_rounds, max_tokens, temperature,
                    router_top_k=2,
                )
            logger.info(f"  A3: initial={results['A3_critic_specialization'].initial_accuracy:.3f} final={results['A3_critic_specialization'].final_accuracy:.3f}")

            # A4: 3 Actors + 4 Critics from FINAL registry, uniform weights (no routing)
            # Uses POST-society-training LoRA — shows joint training + diversity effect.
            # Key difference from A5: uniform_weights=True means all critics
            # contribute equally (no softmax confidence gating)
            logger.info("[A4] 3 Actors + 4 Critics from final registry (no routing, uniform weights)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:3],
                critic_names=all_critic_names,
            )
            results["A4_no_routing"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=4,       # Use ALL critics
                router_uniform=True,  # Equal weights (no softmax)
            )
            logger.info(f"  A4: initial={results['A4_no_routing'].initial_accuracy:.3f} final={results['A4_no_routing'].final_accuracy:.3f}")

            # A5: 3 Actors + 4 Critics from FINAL registry + Router (full system)
            logger.info("[A5] 3 Actors + 4 Critics + Router (full system)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:3],
                critic_names=all_critic_names,
            )
            results["A5_full_system"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=router_top_k,  # From config
                router_uniform=False,       # Softmax confidence weighting
            )
            logger.info(f"  A5: initial={results['A5_full_system'].initial_accuracy:.3f} final={results['A5_full_system'].final_accuracy:.3f}")
        else:
            # Single main evaluation with full system
            logger.info("[Main] Full system evaluation...")
            a_configs, c_configs, lora = _build_agent_configs(registry)
            results["main"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=router_top_k,
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

    # Load society registry
    logger.info("[Step 1] Loading society registry...")
    registry = load_society_registry(args.society_dir)

    # Load dataset
    logger.info("[Step 2] Loading dataset...")
    samples = load_dataset(args.dataset, args.seed, args.max_samples)
    logger.info(f"  Test samples: {len(samples)}")

    # Run all evaluations with shared engine
    logger.info("[Step 3] Running evaluation (shared base model)...")
    total_start = time.time()

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
        device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        run_ablations=args.run_ablations,
        max_model_len=args.max_model_len,
        max_lora_rank=args.max_lora_rank,
        router_top_k=args.router_top_k,
        phase_actor_dir=phase_actor_dir,
        phase_critic_dir=phase_critic_dir,
    )
    total_time = time.time() - total_start
    logger.info(f"Total evaluation time: {total_time:.1f}s")

    # Save results
    logger.info("[Step 4] Saving results...")

    results_file = os.path.join(output_dir, "results.json")
    first_result = list(ablation_results.values())[0] if ablation_results else None
    with open(results_file, "w") as f:
        json.dump({
            "ablation_results": {
                name: {
                    "initial_accuracy": r.initial_accuracy,
                    "final_accuracy": r.final_accuracy,
                    "per_round_accuracy": r.per_round_accuracy,
                    "improvement_rate": r.improvement_rate,
                    "absolute_improvement": r.absolute_improvement,
                    "consensus_accuracy": r.consensus_accuracy,
                    "diversity_score": r.diversity_score,
                    "ci_95": list(r.ci_95),
                    "num_samples": r.num_samples,
                    "eval_time_seconds": r.eval_time_seconds,
                }
                for name, r in ablation_results.items()
            },
            "total_eval_time_seconds": total_time,
            "sample_details": (ablation_results.get("A5_full_system") or first_result).sample_details if (ablation_results.get("A5_full_system") or first_result) else [],
        }, f, indent=2)

    logger.info(f"  Results saved: {results_file}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    for name, result in ablation_results.items():
        logger.info(f"\n[{name}]")
        logger.info(f"  Initial Accuracy:  {result.initial_accuracy:.4f}")
        logger.info(f"  Final Accuracy:    {result.final_accuracy:.4f}")
        logger.info(f"  Improvement Rate:  {result.improvement_rate:.4f}")
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
