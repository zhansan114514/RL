"""
Society trainer: Multi-Agent alternating training scheduler.

Extends ACC-Collab's alternating training from 1 Actor + 1 Critic
to N Actors + M Critics with data-level diversification.

From experiment plan:
  For each iteration:
    Phase A: Fix all Actors -> Train all Critics (each on its error-type data)
    Phase B: Fix all Critics -> Train all Actors (each on its style data)

Preference pairs are generated using the LLM itself (guided vs natural trajectories),
following the ACC-Collab paper's approach, NOT hardcoded template strings.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.society.agent_registry import AgentRegistry, AgentConfig, AgentRole, ReasoningStyle, ErrorType, CRITIC_SPECIALTY_PROMPTS
from src.society.data_classifier import (
    classify_reasoning_style, ClassificationError,
)
from src.society.diversity_split import DiversitySplit
from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer

logger = logging.getLogger(__name__)


def _get_rng(seed: int):
    """Get a numpy random generator for sampling."""
    import numpy as np
    return np.random.default_rng(seed)


@dataclass
class SocietyTrainingResult:
    """Result of society training."""
    actor_paths: dict[str, str]  # actor_name -> checkpoint path
    critic_paths: dict[str, str]  # critic_name -> checkpoint path
    metrics: dict[str, Any] = field(default_factory=dict)


def society_alternating_train(
    registry: AgentRegistry,
    dataset: list[dict],
    dataset_name: str,
    output_base_dir: str = "cache/society",
    num_iterations: int = 1,
    num_rounds: int = 5,
    num_simulations: int = 5,
    reward_threshold: float = 0.0,
    lora_r: int = 256,
    lora_alpha: int = 512,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 1,
    max_length: int = 2048,
    beta: float = 0.1,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    device: int = 0,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
) -> SocietyTrainingResult:
    """
    Train N Actors + M Critics in alternating fashion.

    Each iteration:
      Phase A: Fix all Actors, train each Critic on its error-type subset
      Phase B: Fix all Critics, train each Actor on its reasoning-style subset

    Crash recovery via checkpoint_dir: resumes from last completed phase.

    Args:
        registry: AgentRegistry with all Actors and Critics.
        dataset: Training dataset (list of standardized samples).
        dataset_name: Dataset name (for prompt templates).
        output_base_dir: Base directory for outputs.
        num_iterations: Number of alternating iterations.
        num_rounds: Deliberation rounds for trajectory generation.
        num_simulations: MC roll-out simulations.
        reward_threshold: Preference pair filtering threshold.
        lora_r: LoRA rank (default 256).
        lora_alpha: LoRA alpha (default 512).
        learning_rate: Learning rate (default 5e-5).
        batch_size: Batch size.
        num_epochs: Epochs per agent.
        max_length: Max sequence length.
        beta: DPO beta (default 0.1).
        seed: Random seed.
        checkpoint_dir: Directory for crash recovery.
        device: CUDA device index.
        dtype: Model dtype.
        gpu_memory_utilization: vLLM GPU memory utilization.
        max_model_len: Max model sequence length.

    Returns:
        SocietyTrainingResult with checkpoint paths and metrics.
    """
    actors = registry.list_actors()
    critics = registry.list_critics()

    if not actors or not critics:
        logger.error("Registry must have at least one Actor and one Critic")
        return SocietyTrainingResult({}, {}, {"status": "error", "message": "missing agents"})

    if num_iterations <= 0:
        logger.warning("num_iterations=0, no training will occur")
        return SocietyTrainingResult(
            {a.name: a.lora_path or "" for a in actors},
            {c.name: c.lora_path or "" for c in critics},
            {"status": "skipped", "iterations": 0},
        )

    # Setup checkpointing
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(output_base_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "society_training_state.json"

    # Load existing checkpoint
    state = _load_checkpoint(ckpt_file)
    start_iteration = state.get("iteration", 0)
    actor_paths = state.get("actor_paths", {})
    critic_paths = state.get("critic_paths", {})
    metrics = state.get("metrics", {})

    for iteration in range(start_iteration, num_iterations):
        logger.info(f"=== Society Training Iteration {iteration + 1}/{num_iterations} ===")

        # ---- Phase A: Train all Critics (fix Actors) ----
        logger.info(f"Phase A: Training {len(critics)} Critics (Actors frozen)")

        # Generate trajectories with guided vs natural Critic feedback
        # using ALL actors for diverse reference responses
        logger.info("  Generating deliberation trajectories for Critic training...")
        critic_trajectories, shared_engine = _generate_critic_trajectories(
            actors=actors,
            registry=registry,
            dataset=dataset,
            dataset_name=dataset_name,
            num_rounds=num_rounds,
            device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        )
        logger.info(f"  Generated {len(critic_trajectories)} trajectories for all critics")

        for critic in critics:
            critic_iter_dir = f"{output_base_dir}/critics/{critic.name}/iter_{iteration}"
            Path(critic_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Critic: {critic.display_name} "
                f"(specialty: {critic.error_specialty.value})"
            )

            # Build LLM-generated preference pairs for this critic
            preference_pairs = _build_critic_preference_pairs(
                trajectories=critic_trajectories,
                critic=critic,
                dataset=dataset,
                dataset_name=dataset_name,
                registry=registry,
                device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                seed=seed,
                engine=shared_engine,  # Reuse shared engine from trajectory generation
            )

            if not preference_pairs:
                logger.warning(f"  No preference pairs for {critic.name}, skipping")
                critic_paths[critic.name] = critic_iter_dir
                metrics[f"critic_{critic.name}_iter{iteration}"] = {"status": "skipped", "pairs": 0}
                continue

            # Run DPO training
            logger.info(f"  Training with {len(preference_pairs)} preference pairs")
            checkpoint_path = _run_dpo_training(
                model_name=registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct",
                preference_pairs=preference_pairs,
                output_dir=critic_iter_dir,
                agent_type="critic",
                agent_name=critic.name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                max_length=max_length,
                beta=beta,
                seed=seed,
                device=device,
            )

            critic_paths[critic.name] = checkpoint_path or critic_iter_dir
            metrics[f"critic_{critic.name}_iter{iteration}"] = {
                "status": "completed",
                "pairs": len(preference_pairs),
                "path": checkpoint_path or "",
            }

        # Cleanup shared critic engine before Phase B
        if shared_engine is not None:
            del shared_engine
            shared_engine = None
            _cleanup_gpu()

        # ---- Phase B: Train all Actors (fix Critics) ----
        logger.info(f"Phase B: Training {len(actors)} Actors (Critics frozen)")

        # Build preference pairs for ALL actors with a SINGLE vLLM instance
        all_actor_pairs = _build_all_actor_preference_pairs(
            actors=actors,
            dataset=dataset,
            dataset_name=dataset_name,
            registry=registry,
            device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        )

        for actor in actors:
            actor_iter_dir = f"{output_base_dir}/actors/{actor.name}/iter_{iteration}"
            Path(actor_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Actor: {actor.display_name} "
                f"(style: {actor.reasoning_style.value})"
            )

            preference_pairs = all_actor_pairs.get(actor.name, [])

            if not preference_pairs:
                logger.warning(f"  No preference pairs for {actor.name}, skipping")
                actor_paths[actor.name] = actor_iter_dir
                metrics[f"actor_{actor.name}_iter{iteration}"] = {"status": "skipped", "pairs": 0}
                continue

            # Run DPO training
            logger.info(f"  Training with {len(preference_pairs)} preference pairs")
            checkpoint_path = _run_dpo_training(
                model_name=registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct",
                preference_pairs=preference_pairs,
                output_dir=actor_iter_dir,
                agent_type="actor",
                agent_name=actor.name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                max_length=max_length,
                beta=beta,
                seed=seed,
                device=device,
            )

            actor_paths[actor.name] = checkpoint_path or actor_iter_dir
            metrics[f"actor_{actor.name}_iter{iteration}"] = {
                "status": "completed",
                "pairs": len(preference_pairs),
                "path": checkpoint_path or "",
            }

        # Save checkpoint
        _save_checkpoint(ckpt_file, {
            "iteration": iteration + 1,
            "actor_paths": actor_paths,
            "critic_paths": critic_paths,
            "metrics": metrics,
        })

    # Update registry with new LoRA paths
    for name, path in actor_paths.items():
        agent = registry.get(name)
        if agent:
            agent.lora_path = path
    for name, path in critic_paths.items():
        agent = registry.get(name)
        if agent:
            agent.lora_path = path

    # Save updated registry
    registry.save(Path(output_base_dir) / "registry.json")

    return SocietyTrainingResult(
        actor_paths=actor_paths,
        critic_paths=critic_paths,
        metrics=metrics,
    )


# ============================================================
# Trajectory generation helpers
# ============================================================

def _cleanup_gpu():
    """Force cleanup GPU memory after vLLM unload."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _generate_critic_trajectories(
    actors: list[AgentConfig],
    registry: AgentRegistry,
    dataset: list[dict],
    dataset_name: str,
    num_rounds: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
) -> tuple[list[dict], Any]:
    """Generate deliberation trajectories for Critic training data.

    Loads the base model ONCE, runs deliberation with specialized actors
    and a temporary generic critic (base model without LoRA), and collects
    trajectories. The generic critic provides basic feedback so actors
    engage in genuine deliberation rather than generating in isolation.

    Returns:
        Tuple of (trajectories, engine). The engine is kept alive for reuse
        by downstream preference pair generation.
    """
    from src.inference.vllm_server import VLLMInference
    from src.society.multi_deliberation import multi_agent_deliberate_single_gpu

    trajectories = []
    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"
    engine = None

    # Create a temporary generic critic using the base model (no LoRA)
    temp_critic = AgentConfig(
        name="temp_critic",
        role=AgentRole.CRITIC,
        error_specialty=ErrorType.LOGIC,
        model_path=model_name,
        lora_path=None,  # No LoRA — use base model as generic critic
        system_prompt=(
            "You are a critical reviewer. Analyze the given solution, "
            "identify any errors in reasoning or calculation, and provide "
            "constructive feedback to help reach the correct answer."
        ),
        temperature=0.7,
        max_tokens=256,
    )

    # Determine if any actor has a LoRA path so we can enable LoRA on the engine
    actors_with_lora = [a for a in actors if a.lora_path]
    enable_lora = len(actors_with_lora) > 0
    max_loras = len(actors_with_lora) if enable_lora else 1

    try:
        engine = VLLMInference(
            model_name,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=256,
        )

        # Use a subset for trajectory generation (was 50, too small for 7 agents)
        max_traj = min(len(dataset), 200)
        for i, sample in enumerate(dataset[:max_traj]):
            if (i + 1) % 10 == 0:
                logger.info(f"  Generating trajectory {i + 1}/{max_traj}")

            result = multi_agent_deliberate_single_gpu(
                inference_engine=engine,
                actors=actors,
                critics=[temp_critic],  # Use generic base-model critic
                sample=sample,
                dataset_name=dataset_name,
                num_rounds=min(num_rounds, 2),
                max_tokens=512,
                temperature=0.7,
            )

            trajectories.append({
                "sample": sample,
                "actor_responses": result.final_answers,
                "rounds": [
                    {
                        "actor_response": r.actor_responses,
                        "actor_answers": r.actor_answers,
                    }
                    for r in result.rounds
                ],
            })

        # Keep engine alive — caller is responsible for cleanup

    except Exception as e:
        logger.error(f"Failed to generate trajectories: {e}")
        if engine is not None:
            del engine
            engine = None
        _cleanup_gpu()

    return trajectories, engine


def _build_critic_preference_pairs(
    trajectories: list[dict],
    critic: AgentConfig,
    dataset: list[dict],
    dataset_name: str,
    registry: AgentRegistry,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
    engine: Any = None,
) -> list[dict]:
    """Build DPO preference pairs for a specific Critic using LLM-generated feedback.

    Key improvement: filters error samples by this critic's specialty via
    GLM API classification. Only errors matching the critic's error_specialty
    are used, ensuring true data-level diversification.

    Uses guided (chosen) vs generic (rejected) feedback following the
    ACC-Collab paper's approach.
    """
    from src.society.multi_deliberation import _build_critic_prompt

    preference_pairs = []
    specialty = critic.error_specialty
    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"

    # Collect all wrong-answer samples from trajectories
    all_error_samples = []
    for traj in trajectories:
        sample = traj.get("sample", {})
        correct_answer = sample.get("answer", "")
        task_type = sample.get("task_type", "math")

        for round_data in traj.get("rounds", []):
            actor_responses = round_data.get("actor_responses", {})
            actor_answers = round_data.get("actor_answers", {})

            for actor_name, actor_response in actor_responses.items():
                actor_answer = actor_answers.get(actor_name)
                if actor_answer is None:
                    continue

                # Check if wrong answer
                is_correct = False
                if task_type == "math":
                    is_correct = math_answers_equal(actor_answer or "", correct_answer)
                else:
                    is_correct = normalize_answer(actor_answer, task_type) == normalize_answer(correct_answer, task_type)

                if not is_correct:
                    all_error_samples.append({
                        "sample": sample,
                        "actor_response": actor_response,
                        "actor_answer": actor_answer,
                        "correct_answer": correct_answer,
                        "task_type": task_type,
                    })

    if not all_error_samples:
        logger.warning(f"  No error samples found for {critic.name}")
        return []

    # Use DiversitySplit for error-type classification + balancing
    splitter = DiversitySplit(balance=True, seed=seed, use_api=True)
    error_splits = splitter.split_by_error_type(
        samples=[es["sample"] for es in all_error_samples],
        responses=[es["actor_response"] for es in all_error_samples],
        correct_answers=[es["correct_answer"] for es in all_error_samples],
        extracted_answers=[es["actor_answer"] or "" for es in all_error_samples],
    )

    # Extract balanced samples for this critic's specialty
    specialty_items = error_splits.get(specialty, [])
    filtered_samples = []
    for sample, _response in specialty_items:
        # Find matching error sample by sample identity
        for es in all_error_samples:
            if es["sample"] is sample or (
                es["sample"].get("question") == sample.get("question")
                and es["actor_response"] == _response
            ):
                filtered_samples.append(es)
                break

    logger.info(
        f"  Filtered {len(filtered_samples)}/{len(all_error_samples)} error samples "
        f"for specialty '{specialty.value}' (balanced via DiversitySplit)"
    )

    if not filtered_samples:
        logger.warning(
            f"  No samples matched specialty '{specialty.value}'. "
            f"Skipping this Critic to preserve data-level diversification "
            f"(using all errors would break specialization)."
        )
        return []

    # Use LLM to generate guided (chosen) vs generic (rejected) feedback
    engine_provided = engine is not None
    try:
        if not engine_provided:
            from src.inference.vllm_server import VLLMInference
            engine = VLLMInference(
                model_name,
                cuda_device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                # Critic preference pair generation uses base model only,
                # no LoRA needed — the feedback is generated by prompting.
            )

        for es in filtered_samples:
            sample = es["sample"]
            actor_response = es["actor_response"]

            # Chosen: guided feedback with specialty-specific prompt
            guided_prompt = (
                f"You are reviewing a solution and have identified a {specialty.value} error.\n"
                f"Problem: {sample.get('question', '')}\n"
                f"Student's solution: {actor_response}\n"
                f"Correct answer: {es['correct_answer']}\n\n"
                f"Provide specific, actionable feedback that identifies the {specialty.value} error "
                f"and guides toward the correct solution. "
                f"After your analysis, output your confidence on a scale of 0.0 to 1.0 "
                f"using the format: [Confidence: 0.X]"
            )
            chosen = engine.generate_single(guided_prompt, max_tokens=256, temperature=0.3)

            # Rejected: generic feedback without specialty guidance
            generic_prompt = (
                f"Review this solution briefly.\n"
                f"Problem: {sample.get('question', '')}\n"
                f"Solution: {actor_response}\n\n"
                f"Provide brief feedback. "
                f"After your analysis, output your confidence on a scale of 0.0 to 1.0 "
                f"using the format: [Confidence: 0.X]"
            )
            rejected = engine.generate_single(generic_prompt, max_tokens=256, temperature=0.7)

            if chosen and rejected and chosen.strip() != rejected.strip():
                preference_pairs.append({
                    "sample": sample,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "error_type": specialty.value,
                        "actor_answer": es["actor_answer"],
                        "correct_answer": es["correct_answer"],
                    },
                })

        if not engine_provided:
            del engine
            _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to generate LLM feedback pairs: {e}")
        if not engine_provided and engine is not None:
            del engine
        _cleanup_gpu()

    return preference_pairs


def _build_all_actor_preference_pairs(
    actors: list[AgentConfig],
    dataset: list[dict],
    dataset_name: str,
    registry: AgentRegistry,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
) -> dict[str, list[dict]]:
    """Build DPO preference pairs for ALL Actors with a SINGLE vLLM instance.

    Phase 1: Batch-generate all Actor responses (all actors × all samples).
    Phase 2: Classify styles, filter to actor-matching styles, balance via DiversitySplit.
    Phase 3: Generate rejected responses only for balanced chosen samples.

    This follows the ACC-Collab paper's Algorithm 1 approach.
    """
    from src.inference.vllm_server import VLLMInference
    from src.society.multi_deliberation import _build_actor_prompt

    all_pairs: dict[str, list[dict]] = {actor.name: [] for actor in actors}
    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"
    engine = None

    # Actor preference pair generation doesn't load actor LoRA adapters;
    # it generates fresh responses from the base model and then filters
    # by style.  No LoRA needed here.
    try:
        engine = VLLMInference(
            model_name,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        max_samples = min(len(dataset), 200)

        # Phase 1: Batch-generate all actor responses
        # Each entry: (sample, actor, response, task_type)
        correct_entries: list[tuple[dict, AgentConfig, str, str]] = []

        for si, sample in enumerate(dataset[:max_samples]):
            if (si + 1) % 10 == 0:
                logger.info(f"  Generating actor responses {si + 1}/{max_samples}")

            correct_answer = sample.get("answer", "")
            task_type = sample.get("task_type", "math")

            # Build prompts for ALL actors, then batch in one generate() call
            prompts = [
                _build_actor_prompt(actor, sample, dataset_name, 0, [])
                for actor in actors
            ]
            responses = engine.generate(prompts, max_tokens=512, temperature=0.7)

            for actor, response in zip(actors, responses):
                response = response if isinstance(response, str) else str(response)
                extracted_answer = extract_answer(response, task_type)

                if task_type == "math":
                    is_correct = math_answers_equal(extracted_answer or "", correct_answer)
                else:
                    is_correct = normalize_answer(extracted_answer or "", task_type) == normalize_answer(correct_answer, task_type)

                if is_correct:
                    correct_entries.append((sample, actor, response, task_type))

        logger.info(f"  Collected {len(correct_entries)} correct responses across all actors")

        # Phase 2: Classify styles and filter per actor, then balance
        # Group by actor first
        actor_entries: dict[str, list[tuple[dict, str, str]]] = {a.name: [] for a in actors}
        for sample, actor, response, task_type in correct_entries:
            question = sample.get("question", "")
            correct_answer = sample.get("answer", "")
            style_result = None
            try:
                style_result = classify_reasoning_style(
                    response=response,
                    question=question,
                    correct_answer=correct_answer,
                    use_api=True,
                )
            except ClassificationError:
                pass

            if style_result is not None and style_result.style == actor.reasoning_style:
                actor_entries[actor.name].append((sample, response, task_type))

        # Balance: downsample actors with more samples to match the actor with fewest
        non_empty = {k: v for k, v in actor_entries.items() if v}
        if non_empty:
            target = min(len(v) for v in non_empty.values())
            rng = _get_rng(seed)
            for name in actor_entries:
                entries = actor_entries[name]
                if len(entries) > target:
                    indices = rng.choice(len(entries), size=target, replace=False)
                    actor_entries[name] = [entries[i] for i in indices]
                    logger.info(f"  Balanced {name}: {len(entries)} -> {target}")

        # Phase 3: Generate rejected responses for balanced chosen samples
        for actor in actors:
            for sample, response, task_type in actor_entries[actor.name]:
                question = sample.get("question", "")
                correct_answer = sample.get("answer", "")

                wrong_prompt = (
                    f"{actor.system_prompt}\n\n"
                    f"Solve this problem, but make a common mistake in your reasoning.\n"
                    f"Problem: {question}\n"
                )
                rejected = engine.generate_single(wrong_prompt, max_tokens=512, temperature=0.9)

                if rejected and rejected.strip() != response.strip():
                    rejected_answer = extract_answer(rejected, task_type)
                    if task_type == "math":
                        rejected_is_correct = math_answers_equal(rejected_answer or "", correct_answer)
                    else:
                        rejected_is_correct = normalize_answer(rejected_answer or "", task_type) == normalize_answer(correct_answer, task_type)

                    if not rejected_is_correct:
                        all_pairs[actor.name].append({
                            "sample": sample,
                            "chosen": response,
                            "rejected": rejected,
                            "metadata": {
                                "style": actor.reasoning_style.value,
                            },
                        })

        del engine
        _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to build actor preference pairs: {e}")
        if engine is not None:
            del engine
        _cleanup_gpu()

    for actor in actors:
        logger.info(f"  Actor {actor.name}: {len(all_pairs[actor.name])} preference pairs")

    return all_pairs


# ============================================================
# DPO training helper
# ============================================================

def _run_dpo_training(
    model_name: str,
    preference_pairs: list[dict],
    output_dir: str,
    agent_type: str,
    agent_name: str,
    lora_r: int = 256,
    lora_alpha: int = 512,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 1,
    max_length: int = 2048,
    beta: float = 0.1,
    seed: int = 42,
    device: int = 0,
) -> Optional[str]:
    """Run DPO training as a subprocess (to manage GPU memory cleanly).

    Returns the path to the trained checkpoint, or None on failure.
    """
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)

    # Save preference pairs to a temp file
    pairs_file = os.path.join(output_dir, "preference_pairs.json")
    with open(pairs_file, "w") as f:
        json.dump(preference_pairs, f, ensure_ascii=False, indent=2)

    try:
        from datasets import Dataset
        from src.training.dpo_trainer import train_dpo

        # Convert preference_pairs to HuggingFace Dataset
        hf_data = {
            "prompt": [p.get("sample", {}).get("question", "") for p in preference_pairs],
            "chosen": [p.get("chosen", "") for p in preference_pairs],
            "rejected": [p.get("rejected", "") for p in preference_pairs],
        }
        preference_dataset = Dataset.from_dict(hf_data)

        checkpoint_path = train_dpo(
            model_name_or_path=model_name,
            preference_dataset=preference_dataset,
            output_dir=output_dir,
            model_type=model_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            max_length=max_length,
            beta=beta,
            seed=seed,
            device=device,
        )
        return checkpoint_path

    except Exception as e:
        logger.error(f"DPO training failed for {agent_name}: {e}")
        return None


# ============================================================
# Checkpoint helpers
# ============================================================

def _load_checkpoint(path: Path) -> dict:
    """Load training checkpoint."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_checkpoint(path: Path, data: dict) -> None:
    """Save training checkpoint (atomic)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    logger.info(f"Checkpoint saved: {path}")
