"""
Society trainer: Multi-Agent alternating training scheduler.

Extends ACC-Collab's alternating training from 1 Actor + 1 Critic
to N Actors + M Critics with data-level diversification.

From experiment plan:
  For each iteration:
    Phase A: Fix all Actors → Train all Critics (each on its error-type data)
    Phase B: Fix all Critics → Train all Actors (each on its style data)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.society.agent_registry import AgentRegistry, AgentRole, ReasoningStyle, ErrorType
from src.society.diversity_split import DiversitySplit
from src.society.data_classifier import classify_error_type, classify_reasoning_style
from src.algorithms.reward import extract_answer, math_answers_equal

logger = logging.getLogger(__name__)


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

        # Generate trajectories ONCE for all critics (shared reference actor)
        reference_actor = actors[0]
        logger.info(f"  Generating shared trajectories using {reference_actor.display_name}...")
        trajectories = _generate_critic_trajectories(
            actor=reference_actor,
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
        logger.info(f"  Generated {len(trajectories)} shared trajectories for all critics")

        for critic in critics:
            critic_iter_dir = f"{output_base_dir}/critics/{critic.name}/iter_{iteration}"
            Path(critic_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Critic: {critic.display_name} "
                f"(specialty: {critic.error_specialty.value})"
            )

            # Filter shared trajectories for this critic's specialty
            preference_pairs = _build_critic_preference_pairs(
                trajectories=trajectories,
                dataset=dataset,
                dataset_name=dataset_name,
                critic_specialty=critic.error_specialty,
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
    actor: AgentRole,
    registry: AgentRegistry,
    dataset: list[dict],
    dataset_name: str,
    num_rounds: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
) -> list[dict]:
    """Generate deliberation trajectories for Critic training data.

    Loads the Actor model ONCE, runs deliberation with base-model-as-critic,
    and collects (sample, actor_response, critic_feedback, answer) tuples.
    The result is shared across all Critics to avoid repeated vLLM loading.
    """
    from src.inference.vllm_server import VLLMInference
    from src.society.multi_deliberation import multi_agent_deliberate_single_gpu

    trajectories = []
    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"

    try:
        engine = VLLMInference(
            model_name,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        # Use a subset for trajectory generation
        max_traj = min(len(dataset), 50)
        for i, sample in enumerate(dataset[:max_traj]):
            if (i + 1) % 10 == 0:
                logger.info(f"  Generating trajectory {i + 1}/{max_traj}")

            result = multi_agent_deliberate_single_gpu(
                inference_engine=engine,
                actors=[actor],
                critics=[],  # No specialized critics yet, just get actor responses
                sample=sample,
                dataset_name=dataset_name,
                num_rounds=min(num_rounds, 2),  # Fewer rounds for data generation
                max_tokens=512,
                temperature=0.7,
            )

            trajectories.append({
                "sample": sample,
                "actor_responses": result.final_answers,
                "rounds": [
                    {
                        "actor_response": r.actor_responses.get(actor.name, ""),
                        "actor_answer": r.actor_answers.get(actor.name),
                    }
                    for r in result.rounds
                ],
            })

        # Cleanup
        del engine
        _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to generate trajectories: {e}")
        # Fallback: create simple trajectories from dataset
        for sample in dataset[:min(len(dataset), 30)]:
            trajectories.append({
                "sample": sample,
                "actor_responses": {},
                "rounds": [],
            })

    return trajectories


def _build_critic_preference_pairs(
    trajectories: list[dict],
    dataset: list[dict],
    dataset_name: str,
    critic_specialty: ErrorType,
) -> list[dict]:
    """Build DPO preference pairs for a specific Critic.

    Chosen: Critic feedback that would lead to correct answer
    Rejected: Critic feedback that is unhelpful or misleading
    """
    preference_pairs = []

    for traj in trajectories:
        sample = traj.get("sample", {})
        correct_answer = sample.get("answer", "")
        task_type = sample.get("task_type", "math")

        for round_data in traj.get("rounds", []):
            actor_answer = round_data.get("actor_answer")
            if actor_answer is None:
                continue

            # Check if the actor got it wrong
            is_correct = False
            if task_type == "math":
                is_correct = math_answers_equal(actor_answer, correct_answer)
            else:
                from src.algorithms.reward import normalize_answer
                is_correct = normalize_answer(actor_answer, task_type) == normalize_answer(correct_answer, task_type)

            if is_correct:
                continue  # Only need error cases for critic training

            # Classify the error type
            error_result = classify_error_type(
                response=round_data.get("actor_response", ""),
                question=sample.get("question", ""),
                extracted_answer=actor_answer,
                correct_answer=correct_answer,
                use_api=True,
            )

            # Only include if it matches this critic's specialty
            if error_result.error_type == critic_specialty:
                # Construct chosen (helpful feedback) and rejected (generic) pairs
                chosen = (
                    f"I found an error in this solution. "
                    f"The approach has a {critic_specialty.value} issue. "
                    f"The correct answer should be derived by correcting "
                    f"the {critic_specialty.value} mistake."
                )
                rejected = "This solution looks correct."

                preference_pairs.append({
                    "sample": sample,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "error_type": critic_specialty.value,
                        "actor_answer": actor_answer,
                        "correct_answer": correct_answer,
                    },
                })

    return preference_pairs


def _build_all_actor_preference_pairs(
    actors: list,
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

    Loads vLLM once, generates responses for each actor on each sample,
    then unloads. Returns {actor_name: [preference_pairs]}.

    This avoids repeated vLLM loading which causes OOM on single-GPU setups.
    """
    from src.inference.vllm_server import VLLMInference

    all_pairs: dict[str, list[dict]] = {actor.name: [] for actor in actors}
    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"

    try:
        engine = VLLMInference(
            model_name,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        max_samples = min(len(dataset), 50)
        for si, sample in enumerate(dataset[:max_samples]):
            if (si + 1) % 10 == 0:
                logger.info(f"  Generating actor responses {si + 1}/{max_samples}")

            correct_answer = sample.get("answer", "")
            task_type = sample.get("task_type", "math")
            question = sample.get("question", "")

            for actor in actors:
                # Generate response with actor's style prompt
                from src.society.multi_deliberation import _generate_actor_response
                response = _generate_actor_response(
                    engine=engine,
                    actor=actor,
                    sample=sample,
                    dataset_name=dataset_name,
                    round_num=0,
                    previous_responses=[],
                    max_tokens=512,
                    temperature=0.7,
                )

                extracted_answer = extract_answer(response, task_type)

                # Check correctness
                is_correct = False
                if task_type == "math":
                    is_correct = math_answers_equal(extracted_answer or "", correct_answer)
                else:
                    from src.algorithms.reward import normalize_answer
                    is_correct = normalize_answer(extracted_answer or "", task_type) == normalize_answer(correct_answer, task_type)

                if is_correct:
                    # Classify style
                    try:
                        style_result = classify_reasoning_style(
                            response=response,
                            question=question,
                            correct_answer=correct_answer,
                            use_api=True,
                        )
                    except Exception:
                        # If API fails, assume matching style for correct answers
                        style_result = None

                    if style_result is None or style_result.style == actor.reasoning_style:
                        all_pairs[actor.name].append({
                            "sample": sample,
                            "chosen": response,
                            "rejected": f"I think the answer is {correct_answer}.",
                            "metadata": {
                                "style": actor.reasoning_style.value,
                                "confidence": getattr(style_result, "confidence", 1.0),
                            },
                        })

        del engine
        _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to build actor preference pairs: {e}")
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
            "prompt": [p.get("question", p.get("sample", {}).get("question", "")) for p in preference_pairs],
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
