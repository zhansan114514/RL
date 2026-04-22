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
    max_samples: int = 200,
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

        for critic in critics:
            critic_iter_dir = f"{output_base_dir}/critics/{critic.name}/iter_{iteration}"
            Path(critic_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Critic: {critic.display_name} "
                f"(specialty: {critic.error_specialty.value})"
            )

            # Generate preference pairs using Algorithm 1 from trajectory.py
            preference_pairs = _generate_critic_pairs_algorithm1(
                actors=actors,
                critic=critic,
                dataset=dataset,
                dataset_name=dataset_name,
                registry=registry,
                num_rounds=num_rounds,
                num_simulations=num_simulations,
                reward_threshold=reward_threshold,
                max_samples=max_samples,
                seed=seed,
                device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
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

        _cleanup_gpu()

        # ---- Phase B: Train all Actors (fix Critics) ----
        logger.info(f"Phase B: Training {len(actors)} Actors (Critics frozen)")

        for actor in actors:
            actor_iter_dir = f"{output_base_dir}/actors/{actor.name}/iter_{iteration}"
            Path(actor_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Actor: {actor.display_name} "
                f"(style: {actor.reasoning_style.value})"
            )

            # Generate preference pairs using Algorithm 1 from trajectory.py
            preference_pairs = _generate_actor_pairs_algorithm1(
                actor=actor,
                critics=critics,
                dataset=dataset,
                dataset_name=dataset_name,
                registry=registry,
                num_rounds=num_rounds,
                num_simulations=num_simulations,
                reward_threshold=reward_threshold,
                max_samples=max_samples,
                seed=seed,
                device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )

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
# LoRA Model Adapter — bridges shared VLLMInference + LoRA to
# the interface expected by trajectory.py's generate_trajectories()
# ============================================================

class _LoRAModelAdapter:
    """Wraps a shared VLLMInference engine with an optional LoRA adapter.

    Provides the ``generate()`` and ``generate_single()`` interface that
    ``generate_trajectories`` (trajectory.py) and ``deliberate()``
    (deliberation.py) expect, while routing generation calls through
    the single shared engine with per-agent LoRA requests.
    """

    def __init__(self, engine: Any, lora_request: Any = None):
        self._engine = engine
        self._lora_request = lora_request

    def generate(self, prompts, max_tokens=256, temperature=0.7, **kwargs):
        if self._lora_request is not None:
            return self._generate_with_lora(prompts, max_tokens, temperature)
        return self._engine.generate(
            prompts, max_tokens=max_tokens, temperature=temperature, **kwargs,
        )

    def generate_single(self, prompt, max_tokens=256, temperature=0.7, **kwargs):
        results = self.generate(
            [prompt], max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        return results[0] if results else ""

    def _generate_with_lora(self, prompts, max_tokens, temperature):
        if hasattr(self._engine, "generate_with_lora"):
            return self._engine.generate_with_lora(
                prompts, self._lora_request,
                max_tokens=max_tokens, temperature=temperature,
            )
        # Fallback: vLLM LoRA API directly
        if hasattr(self._engine, "_llm") and self._engine._llm is not None:
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=max_tokens, temperature=temperature, n=1)
            outputs = self._engine._llm.generate(
                prompts, params, lora_request=self._lora_request,
            )
            return [c.text for o in outputs for c in o.outputs]
        raise RuntimeError(
            "Engine does not support LoRA generation. "
            "Create VLLMInference with enable_lora=True."
        )


def _build_lora_adapters(
    engine: Any,
    agents: list[AgentConfig],
) -> dict[str, _LoRAModelAdapter]:
    """Create _LoRAModelAdapter for each agent (uses base model if no LoRA)."""
    from src.society.multi_deliberation import _load_lora_adapter

    adapters: dict[str, _LoRAModelAdapter] = {}
    for agent in agents:
        lora_req = None
        if agent.lora_path:
            try:
                lora_req = _load_lora_adapter(engine, agent.lora_path)
            except Exception as e:
                logger.warning(
                    f"Could not load LoRA for {agent.name}: {e}. "
                    f"Using base model."
                )
        adapters[agent.name] = _LoRAModelAdapter(engine, lora_req)
    return adapters


# ============================================================
# Phase A: Critic preference pairs via Algorithm 1
# ============================================================

def _generate_critic_pairs_algorithm1(
    actors: list[AgentConfig],
    critic: AgentConfig,
    dataset: list[dict],
    dataset_name: str,
    registry: AgentRegistry,
    num_rounds: int,
    num_simulations: int,
    reward_threshold: float,
    max_samples: int,
    seed: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> list[dict]:
    """Generate Critic DPO preference pairs using Algorithm 1 from trajectory.py.

    Pairs the target critic with a reference actor (first with LoRA, or first
    overall) and calls ``generate_trajectories`` for each sample to obtain:
      - Natural deliberation trajectory
      - Guided-toward-correct trajectory
      - Guided-toward-wrong trajectory
      - MC roll-out reward estimation
      - Preference pairs filtered by reward delta >= epsilon

    The critic preference pairs use:
      chosen  = positive_critic (guided toward correct answer)
      rejected = negative_critic (natural / guided toward wrong answer)

    Error-type filtering via DiversitySplit ensures data-level diversification.
    """
    from src.inference.vllm_server import VLLMInference
    from src.algorithms.trajectory import generate_trajectories

    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"
    specialty = critic.error_specialty

    # Pick reference actor (prefer one with LoRA)
    ref_actor = next((a for a in actors if a.lora_path), actors[0])

    # Determine LoRA requirements
    all_agents = [ref_actor, critic]
    enable_lora = any(a.lora_path for a in all_agents)
    max_loras = sum(1 for a in all_agents if a.lora_path) if enable_lora else 0

    n_samples = min(len(dataset), max_samples)
    preference_pairs: list[dict] = []

    try:
        engine = VLLMInference(
            model_name,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora or None,
            max_loras=max(max_loras, 1),
            max_lora_rank=256,
        )

        adapters = _build_lora_adapters(engine, all_agents)
        actor_adapter = adapters[ref_actor.name]
        critic_adapter = adapters[critic.name]

        # Step 1: Run Algorithm 1 for each sample
        raw_pairs: list[dict] = []
        for i, sample in enumerate(dataset[:n_samples]):
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{critic.name}] Algorithm 1: sample {i + 1}/{n_samples}"
                )

            task_type = sample.get("task_type", "math")
            correct_answer = sample.get("answer", "")

            algo_pairs = generate_trajectories(
                actor_model=actor_adapter,
                critic_model=critic_adapter,
                sample=sample,
                dataset_name=dataset_name,
                num_rounds=num_rounds,
                num_simulations=num_simulations,
                reward_threshold=reward_threshold,
                seed=seed + i,
            )

            # Extract critic preference pairs from Algorithm 1 results
            for pair in algo_pairs:
                # pair has: positive, negative (actor), positive_critic, negative_critic
                # For critic DPO: chosen = guided-correct feedback, rejected = natural/wrong feedback
                negative_actor = pair["negative"]
                negative_answer = extract_answer(negative_actor, task_type)

                # Only keep pairs where the actor's response was wrong (error scenario)
                if task_type == "math":
                    is_wrong = not math_answers_equal(
                        negative_answer or "", correct_answer,
                    )
                else:
                    is_wrong = normalize_answer(
                        negative_answer or "", task_type,
                    ) != normalize_answer(correct_answer, task_type)

                if is_wrong:
                    raw_pairs.append({
                        "sample": sample,
                        "chosen": pair["positive_critic"],
                        "rejected": pair["negative_critic"],
                        "actor_response": negative_actor,
                        "actor_answer": negative_answer,
                        "correct_answer": correct_answer,
                        "task_type": task_type,
                        "delta": pair["delta"],
                        "direction": pair["direction"],
                    })

        logger.info(
            f"  [{critic.name}] Algorithm 1 produced {len(raw_pairs)} raw critic pairs"
        )

        # Step 2: Filter by error type via DiversitySplit
        if raw_pairs:
            splitter = DiversitySplit(balance=True, seed=seed, use_api=True)
            error_splits = splitter.split_by_error_type(
                samples=[p["sample"] for p in raw_pairs],
                responses=[p["actor_response"] for p in raw_pairs],
                correct_answers=[p["correct_answer"] for p in raw_pairs],
                extracted_answers=[p["actor_answer"] or "" for p in raw_pairs],
            )

            specialty_items = error_splits.get(specialty, [])
            for sample, _response in specialty_items:
                for p in raw_pairs:
                    if (
                        p["sample"].get("question") == sample.get("question")
                        and p["actor_response"] == _response
                    ):
                        preference_pairs.append({
                            "sample": p["sample"],
                            "chosen": p["chosen"],
                            "rejected": p["rejected"],
                            "metadata": {
                                "error_type": specialty.value,
                                "delta": p["delta"],
                                "direction": p["direction"],
                            },
                        })
                        break

        logger.info(
            f"  [{critic.name}] {len(preference_pairs)}/{len(raw_pairs)} pairs "
            f"matched specialty '{specialty.value}'"
        )

        del engine
        _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to generate critic pairs for {critic.name}: {e}")
        _cleanup_gpu()

    return preference_pairs


# ============================================================
# Phase B: Actor preference pairs via Algorithm 1
# ============================================================

def _generate_actor_pairs_algorithm1(
    actor: AgentConfig,
    critics: list[AgentConfig],
    dataset: list[dict],
    dataset_name: str,
    registry: AgentRegistry,
    num_rounds: int,
    num_simulations: int,
    reward_threshold: float,
    max_samples: int,
    seed: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> list[dict]:
    """Generate Actor DPO preference pairs using Algorithm 1 from trajectory.py.

    Pairs the target actor with a reference critic (first with LoRA, or first
    overall) and calls ``generate_trajectories`` for each sample to obtain
    full Algorithm 1 preference pairs with MC roll-out reward estimation.

    The actor preference pairs use:
      chosen  = positive (guided-toward-correct actor response)
      rejected = negative (natural / guided-toward-wrong actor response)

    Reasoning-style filtering ensures data-level diversification.
    """
    from src.inference.vllm_server import VLLMInference
    from src.algorithms.trajectory import generate_trajectories

    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"

    # Pick reference critic (prefer one with LoRA)
    ref_critic = next((c for c in critics if c.lora_path), critics[0])

    # Determine LoRA requirements
    all_agents = [actor, ref_critic]
    enable_lora = any(a.lora_path for a in all_agents)
    max_loras = sum(1 for a in all_agents if a.lora_path) if enable_lora else 0

    n_samples = min(len(dataset), max_samples)
    preference_pairs: list[dict] = []

    try:
        engine = VLLMInference(
            model_name,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora or None,
            max_loras=max(max_loras, 1),
            max_lora_rank=256,
        )

        adapters = _build_lora_adapters(engine, all_agents)
        actor_adapter = adapters[actor.name]
        critic_adapter = adapters[ref_critic.name]

        # Step 1: Run Algorithm 1 for each sample
        raw_pairs: list[dict] = []
        for i, sample in enumerate(dataset[:n_samples]):
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{actor.name}] Algorithm 1: sample {i + 1}/{n_samples}"
                )

            algo_pairs = generate_trajectories(
                actor_model=actor_adapter,
                critic_model=critic_adapter,
                sample=sample,
                dataset_name=dataset_name,
                num_rounds=num_rounds,
                num_simulations=num_simulations,
                reward_threshold=reward_threshold,
                seed=seed + i,
            )

            # Extract actor preference pairs from Algorithm 1 results
            for pair in algo_pairs:
                raw_pairs.append({
                    "sample": sample,
                    "chosen": pair["positive"],
                    "rejected": pair["negative"],
                    "delta": pair["delta"],
                    "direction": pair["direction"],
                })

        logger.info(
            f"  [{actor.name}] Algorithm 1 produced {len(raw_pairs)} raw actor pairs"
        )

        # Step 2: Filter by reasoning style
        if raw_pairs:
            for p in raw_pairs:
                sample = p["sample"]
                question = sample.get("question", "")
                correct_answer = sample.get("answer", "")
                chosen = p["chosen"]

                style_result = None
                try:
                    style_result = classify_reasoning_style(
                        response=chosen,
                        question=question,
                        correct_answer=correct_answer,
                        use_api=True,
                    )
                except ClassificationError:
                    pass

                if style_result is not None and style_result.style == actor.reasoning_style:
                    preference_pairs.append({
                        "sample": sample,
                        "chosen": chosen,
                        "rejected": p["rejected"],
                        "metadata": {
                            "style": actor.reasoning_style.value,
                            "delta": p["delta"],
                            "direction": p["direction"],
                        },
                    })

        logger.info(
            f"  [{actor.name}] {len(preference_pairs)}/{len(raw_pairs)} pairs "
            f"matched style '{actor.reasoning_style.value}'"
        )

        del engine
        _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to generate actor pairs for {actor.name}: {e}")
        _cleanup_gpu()

    return preference_pairs


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
