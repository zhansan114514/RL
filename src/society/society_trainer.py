"""
Society trainer: Multi-Agent alternating training scheduler.

Extends ACC-Collab's alternating training from 1 Actor + 1 Critic
to N Actors + M Critics with data-level diversification.

For each iteration:
  Phase A: Fix all Actors -> Train each Critic on a mixture of general,
           specialty, and calibration data routed by multi-dimensional error profiles.
  Phase B: Fix all Critics -> Train each Actor on its reasoning-style subset.

Data routing uses error-profile classification to construct critic-specific
mixture datasets (general/specialty/calibration).  The routing_weight stored
in metadata reflects the mixture sampling probability — it is NOT applied
directly to the DPO loss.  Actual DPO training uses unweighted loss on the
mixture dataset.

Preference pairs are generated using the LLM itself (guided vs natural trajectories),
following the ACC-Collab paper's approach, NOT hardcoded template strings.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.society.agent_registry import AgentRegistry, AgentConfig, ReasoningStyle
from src.society.data_classifier import (
    classify_reasoning_style, ClassificationError,
)
from src.society.diversity_split import DiversitySplit, summarize_critic_training_pairs
from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer

logger = logging.getLogger(__name__)


def _cleanup_gpu():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _create_phase_engine(
    model_name: str,
    agents: list[AgentConfig],
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_concurrent_loras: int | None = None,
):
    """Create a shared VLLMInference engine for all agents in a training phase.

    The engine is configured to accommodate LoRA adapters for all agents
    that have lora_path set.

    Args:
        max_concurrent_loras: Override for max_loras.  When *None*, defaults to
            the number of agents that have a LoRA path (one slot per agent).
            Set this to a smaller number (e.g. 2) when only a subset of agents
            are active concurrently, to reduce GPU memory reservation.
    """
    from src.inference.vllm_server import VLLMInference

    enable_lora = any(a.lora_path for a in agents)

    if max_concurrent_loras is not None:
        max_loras = max_concurrent_loras
    else:
        max_loras = sum(1 for a in agents if a.lora_path) if enable_lora else 0

    return VLLMInference(
        model_name,
        cuda_device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=enable_lora or None,
        max_loras=max(max_loras, 1),
        max_lora_rank=256,
    )


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


def _sync_lora_paths(agents: list[AgentConfig], paths: dict[str, str]) -> None:
    """Apply the latest known LoRA paths to in-memory agent configs."""
    for agent in agents:
        path = paths.get(agent.name)
        if path:
            agent.lora_path = path


def society_alternating_train(
    registry: AgentRegistry,
    dataset: list[dict],
    dataset_name: str,
    output_base_dir: str = "cache/society",
    num_iterations: int = 1,
    num_rounds: int = 5,
    num_simulations: int = 5,
    trajectory_max_tokens: int = 256,
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
    min_pairs_per_critic: int = 64,
    min_specialty_items: int = 32,
    min_specialty_ratio: float = 0.08,
    specialty_ratio: float = 0.7,
    general_ratio: float = 0.2,
    calibration_ratio: float = 0.1,
    classifications_cache_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
    strict_classification: bool = True,
    max_classification_failure_rate: float = 0.0,
    max_classification_workers: int = 4,
    request_timeout: int | float = 30,
    max_retries: int = 5,
    retry_delay: int | float = 5,
) -> SocietyTrainingResult:
    """
    Train N Actors + M Critics in alternating fashion.

    Each iteration:
      Phase A: Fix all Actors -> Train each Critic on a mixture of general,
               specialty, and calibration data routed by error profiles.
      Phase B: Fix all Critics -> Train each Actor on its reasoning-style subset.

    Data routing constructs critic-specific mixture datasets via error-profile
    classification.  The routing_weight in metadata reflects mixture sampling
    probability, NOT a direct DPO loss weighting.

    Crash recovery via checkpoint_dir: resumes from last completed phase.

    Args:
        registry: AgentRegistry with all Actors and Critics.
        dataset: Training dataset (list of standardized samples).
        dataset_name: Dataset name (for prompt templates).
        output_base_dir: Base directory for outputs.
        num_iterations: Number of alternating iterations.
        num_rounds: Deliberation rounds for trajectory generation.
        num_simulations: MC roll-out simulations.
        trajectory_max_tokens: Max tokens for Algorithm 1 trajectory generation.
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
        min_pairs_per_critic: Minimum selected DPO pairs required to train a critic.
        min_specialty_items: Minimum target-skill routed items required to activate a critic.
        min_specialty_ratio: Minimum target-skill share required to activate a critic.

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
    # Resume granularity: skip agents whose phase already completed in a
    # partially-run iteration.  Keys: "phase_A_done" / "phase_B_done" ->
    # list of agent names that finished successfully.
    phase_done: dict[str, list[str]] = state.get("phase_done", {})

    _sync_lora_paths(actors, actor_paths)
    _sync_lora_paths(critics, critic_paths)

    for iteration in range(start_iteration, num_iterations):
        logger.info(f"=== Society Training Iteration {iteration + 1}/{num_iterations} ===")

        model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"

        # ---- Phase A: Train all Critics (fix Actors) ----
        logger.info(
            f"Phase A: Training {len(critics)} Critics "
            f"(Actors frozen, error-profile mixture routing)"
        )

        # Per-critic engine: create a fresh engine for each critic to avoid
        # GPU memory accumulation across agents.  Each engine is destroyed
        # before DPO training (which loads its own model).
        phase_a_engine = None

        phase_a_done = set(phase_done.get("phase_A", []))

        for critic in critics:
            # Skip if this critic already completed in a resumed run
            if critic.name in phase_a_done:
                logger.info(
                    f"  Skipping Critic {critic.display_name} (already completed)"
                )
                continue

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
                max_tokens=trajectory_max_tokens,
                reward_threshold=reward_threshold,
                max_samples=max_samples,
                seed=seed,
                device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                engine=None,
                classifications_cache_dir=classifications_cache_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                strict_classification=strict_classification,
                max_classification_failure_rate=max_classification_failure_rate,
                max_classification_workers=max_classification_workers,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                min_specialty_items=min_specialty_items,
                min_specialty_ratio=min_specialty_ratio,
                specialty_ratio=specialty_ratio,
                general_ratio=general_ratio,
                calibration_ratio=calibration_ratio,
            )

            if len(preference_pairs) < min_pairs_per_critic:
                logger.warning(
                    f"  Skipping {critic.name}: {len(preference_pairs)} pairs < "
                    f"{min_pairs_per_critic} minimum"
                )
                critic_paths[critic.name] = critic.lora_path or ""
                metrics[f"critic_{critic.name}_iter{iteration}"] = {
                    "status": "skipped",
                    "pairs": len(preference_pairs),
                    "reason": "below_min_pairs_per_critic",
                    "min_pairs_per_critic": min_pairs_per_critic,
                    "critic_training_metrics": summarize_critic_training_pairs(preference_pairs),
                }
                phase_a_done.add(critic.name)
                _save_checkpoint(ckpt_file, {
                    "iteration": iteration,
                    "actor_paths": actor_paths,
                    "critic_paths": critic_paths,
                    "metrics": metrics,
                    "phase_done": {
                        **phase_done,
                        "phase_A": sorted(phase_a_done),
                    },
                })
                continue

            # Destroy vLLM engine before DPO to free GPU memory.
            # The engine will be recreated on the next iteration if needed
            # (when engine=None is passed, the generation function creates
            # and manages its own engine).
            if phase_a_engine is not None:
                del phase_a_engine
                phase_a_engine = None
            _cleanup_gpu()

            # Run DPO training
            logger.info(f"  Training with {len(preference_pairs)} preference pairs")
            checkpoint_path = _run_dpo_training(
                model_name=registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct",
                preference_pairs=preference_pairs,
                output_dir=critic_iter_dir,
                agent_type="critic",
                agent_name=critic.name,
                dataset_name=dataset_name,
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

            if checkpoint_path is not None:
                critic_paths[critic.name] = checkpoint_path
                metrics[f"critic_{critic.name}_iter{iteration}"] = {
                    "status": "completed",
                    "pairs": len(preference_pairs),
                    "path": checkpoint_path,
                    "critic_training_metrics": summarize_critic_training_pairs(preference_pairs),
                }
            else:
                logger.warning(
                    f"  DPO training failed for {critic.name}, "
                    f"keeping previous LoRA path"
                )
                metrics[f"critic_{critic.name}_iter{iteration}"] = {
                    "status": "failed",
                    "pairs": len(preference_pairs),
                    "path": critic_paths.get(critic.name, ""),
                    "critic_training_metrics": summarize_critic_training_pairs(preference_pairs),
                }

            # Per-critic checkpoint: if we crash after this point, the next
            # run will skip this critic and start from the next one.
            _save_checkpoint(ckpt_file, {
                "iteration": iteration,
                "actor_paths": actor_paths,
                "critic_paths": critic_paths,
                "metrics": metrics,
                "phase_done": {
                    **phase_done,
                    "phase_A": [
                        n for n, p in critic_paths.items()
                        if metrics.get(f"critic_{n}_iter{iteration}", {}).get("status") == "completed"
                    ],
                },
            })

        # Cleanup Phase A engine (may already be None if DPO was run)
        if phase_a_engine is not None:
            del phase_a_engine
        _cleanup_gpu()

        _sync_lora_paths(critics, critic_paths)

        # ---- Phase B: Train all Actors (fix Critics) ----
        logger.info(f"Phase B: Training {len(actors)} Actors (Critics frozen)")

        # Per-actor engine: create fresh engine for each actor to avoid
        # GPU memory accumulation across agents.
        phase_b_engine = None

        phase_b_done = set(phase_done.get("phase_B", []))

        for actor in actors:
            # Skip if this actor already completed in a resumed run
            if actor.name in phase_b_done:
                logger.info(
                    f"  Skipping Actor {actor.display_name} (already completed)"
                )
                continue

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
                max_tokens=trajectory_max_tokens,
                reward_threshold=reward_threshold,
                max_samples=max_samples,
                seed=seed,
                device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                engine=None,
                classifications_cache_dir=classifications_cache_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                strict_classification=strict_classification,
                max_classification_failure_rate=max_classification_failure_rate,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

            if not preference_pairs:
                logger.warning(f"  No preference pairs for {actor.name}, skipping")
                metrics[f"actor_{actor.name}_iter{iteration}"] = {"status": "skipped", "pairs": 0}
                phase_b_done.add(actor.name)
                _save_checkpoint(ckpt_file, {
                    "iteration": iteration,
                    "actor_paths": actor_paths,
                    "critic_paths": critic_paths,
                    "metrics": metrics,
                    "phase_done": {
                        **phase_done,
                        "phase_A": phase_done.get("phase_A", []),
                        "phase_B": sorted(phase_b_done),
                    },
                })
                continue

            # Destroy vLLM engine before DPO to free GPU memory.
            if phase_b_engine is not None:
                del phase_b_engine
                phase_b_engine = None
            _cleanup_gpu()

            # Run DPO training
            logger.info(f"  Training with {len(preference_pairs)} preference pairs")
            checkpoint_path = _run_dpo_training(
                model_name=registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct",
                preference_pairs=preference_pairs,
                output_dir=actor_iter_dir,
                agent_type="actor",
                agent_name=actor.name,
                dataset_name=dataset_name,
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

            if checkpoint_path is not None:
                actor_paths[actor.name] = checkpoint_path
                metrics[f"actor_{actor.name}_iter{iteration}"] = {
                    "status": "completed",
                    "pairs": len(preference_pairs),
                    "path": checkpoint_path,
                }
            else:
                logger.warning(
                    f"  DPO training failed for {actor.name}, "
                    f"keeping previous LoRA path"
                )
                metrics[f"actor_{actor.name}_iter{iteration}"] = {
                    "status": "failed",
                    "pairs": len(preference_pairs),
                    "path": actor_paths.get(actor.name, ""),
                }

            # Per-actor checkpoint
            _save_checkpoint(ckpt_file, {
                "iteration": iteration,
                "actor_paths": actor_paths,
                "critic_paths": critic_paths,
                "metrics": metrics,
                "phase_done": {
                    **phase_done,
                    "phase_A": phase_done.get("phase_A", []),
                    "phase_B": [
                        n for n, p in actor_paths.items()
                        if metrics.get(f"actor_{n}_iter{iteration}", {}).get("status") == "completed"
                    ],
                },
            })

        # Cleanup Phase B engine (may already be None if DPO was run)
        if phase_b_engine is not None:
            del phase_b_engine
        _cleanup_gpu()

        _sync_lora_paths(actors, actor_paths)

        # End-of-iteration checkpoint (iteration is now complete, so bump
        # the counter and clear phase_done for the next iteration).
        _save_checkpoint(ckpt_file, {
            "iteration": iteration + 1,
            "actor_paths": actor_paths,
            "critic_paths": critic_paths,
            "metrics": metrics,
            "phase_done": {},
        })

    # Update registry with new LoRA paths (only if training succeeded)
    for name, path in actor_paths.items():
        agent = registry.get(name)
        if agent and path:
            agent.lora_path = path
    for name, path in critic_paths.items():
        agent = registry.get(name)
        if agent and path:
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
    """Create _LoRAModelAdapter for each agent.

    Agents with ``lora_path`` must successfully load that adapter.  Falling
    back to the base model would invalidate multi-agent experiments because
    training logs could claim an agent participated while its specialized
    weights were never used.
    """
    from src.society.multi_deliberation import LoRAError, _load_lora_adapter

    adapters: dict[str, _LoRAModelAdapter] = {}
    for agent in agents:
        lora_req = None
        if agent.lora_path:
            try:
                lora_req = _load_lora_adapter(engine, agent.lora_path)
            except LoRAError as e:
                raise LoRAError(
                    f"Required LoRA adapter for agent '{agent.name}' failed "
                    f"to load from '{agent.lora_path}': {e}"
                ) from e
            if lora_req is None:
                raise LoRAError(
                    f"Required LoRA adapter for agent '{agent.name}' at "
                    f"'{agent.lora_path}' produced no LoRARequest."
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
    max_tokens: int,
    reward_threshold: float,
    max_samples: int,
    seed: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    engine: Any = None,
    classifications_cache_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
    strict_classification: bool = True,
    max_classification_failure_rate: float = 0.0,
    max_classification_workers: int = 4,
    request_timeout: int | float = 30,
    max_retries: int = 5,
    retry_delay: int | float = 5,
    min_specialty_items: int = 32,
    min_specialty_ratio: float = 0.08,
    specialty_ratio: float = 0.7,
    general_ratio: float = 0.2,
    calibration_ratio: float = 0.1,
) -> list[dict]:
    """Generate Critic DPO preference pairs using Algorithm 1 from trajectory.py.

    Round-robins across all available actors (one per sample) so the critic
    learns to give feedback for diverse actor styles, matching the multi-actor
    inference setting.  Each ``generate_trajectories`` call still uses only
    1 actor + 1 critic LoRA concurrently.

    The critic preference pairs use:
      chosen  = positive_critic (guided toward correct answer)
      rejected = negative_critic (natural / guided toward wrong answer)

    Error-profile routing via DiversitySplit constructs a critic-specific mixture
    dataset (general/specialty/calibration).  The routing_weight stored in
    metadata reflects mixture sampling probability, NOT a direct DPO loss
    weighting — actual DPO training uses unweighted loss on the mixture.
    """
    from src.inference.vllm_server import VLLMInference
    from src.algorithms.deliberation import deliberate_batch
    from src.algorithms.trajectory import _generate_guided_pairs_for_batch, generate_wrong_answer

    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"
    specialty = critic.error_specialty

    all_agents = list(actors) + [critic]

    n_samples = min(len(dataset), max_samples)
    preference_pairs: list[dict] = []
    _owns_engine = engine is None

    try:
        if _owns_engine:
            enable_lora = any(a.lora_path for a in all_agents)
            max_loras = sum(1 for a in all_agents if a.lora_path) if enable_lora else 0
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
        critic_adapter = adapters[critic.name]

        actors_with_lora = [a for a in actors if a.lora_path]
        actors_pool = actors_with_lora if actors_with_lora else actors

        samples_subset = dataset[:n_samples]

        # Group samples by their round-robin actor for batched deliberation.
        actor_groups: dict[str, list[tuple[int, dict]]] = {}
        for i, sample in enumerate(samples_subset):
            ref_actor = actors_pool[i % len(actors_pool)]
            actor_groups.setdefault(ref_actor.name, []).append((i, sample))

        raw_pairs: list[dict] = []

        for actor_name, indexed_samples in actor_groups.items():
            actor_adapter = adapters[actor_name]
            group_samples = [s for _, s in indexed_samples]

            logger.info(
                f"  [{critic.name}] Batched with actor {actor_name}: "
                f"{len(group_samples)} samples"
            )

            # Batched natural deliberation (biggest speedup: N*2*T → 2*T calls)
            trajectories = deliberate_batch(
                actor_adapter, critic_adapter, group_samples, dataset_name,
                num_rounds=num_rounds, max_tokens=max_tokens, temperature=0.7,
            )

            correct_answers: list[str] = []
            wrong_answers: list[str] = []
            for j, sample in enumerate(group_samples):
                rng = random.Random(seed + indexed_samples[j][0])
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                correct_answers.append(correct_answer)
                wrong_answers.append(generate_wrong_answer(
                    correct_answer, sample.get("choices"),
                    task_type=task_type, rng=rng,
                ))

            algo_pairs = _generate_guided_pairs_for_batch(
                actor_adapter, critic_adapter, group_samples, trajectories,
                dataset_name, correct_answers, wrong_answers,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            logger.info(
                f"  [{critic.name}] Guided+MC batch with actor {actor_name}: "
                f"{len(algo_pairs)} Algorithm 1 pairs"
            )

            for pair in algo_pairs:
                sample = pair["sample"]
                negative_actor = pair["negative"]
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                negative_answer = extract_answer(negative_actor, task_type)

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

        # Step 2: Route by error profile, then build an adaptive specialist mix.
        if raw_pairs:
            # Try to use pre-classified data from phase 2 to ensure
            # single source of truth across all training phases
            pre_classified = os.path.join(
                classifications_cache_dir, "classified_data.json"
            )
            splitter = DiversitySplit(
                balance=False, seed=seed, use_api=True,
                cache_dir=classifications_cache_dir,
                pre_classified_file=pre_classified if os.path.exists(pre_classified) else None,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                strict_classification=strict_classification,
                max_classification_failure_rate=max_classification_failure_rate,
                max_classification_workers=max_classification_workers,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            routed_items = splitter.split_by_error_profile(
                samples=[p["sample"] for p in raw_pairs],
                responses=[p["actor_response"] for p in raw_pairs],
                correct_answers=[p["correct_answer"] for p in raw_pairs],
                extracted_answers=[p["actor_answer"] or "" for p in raw_pairs],
                dataset_name=dataset_name,
            )

            if specialty is None:
                raise ValueError(f"Critic '{critic.name}' has no error_specialty set")

            raw_skill_dist = Counter(
                item.skill.value if item.skill else "general"
                for item in routed_items
            )
            logger.info(
                f"  [{critic.name}] raw routed profile distribution: "
                f"{dict(raw_skill_dist)}"
            )

            critic_items = splitter.build_critic_training_mix(
                all_items=routed_items,
                target_skill=specialty,
                max_items=max_samples,
                min_specialty_items=min_specialty_items,
                min_specialty_ratio=min_specialty_ratio,
                specialty_ratio=specialty_ratio,
                general_ratio=general_ratio,
                calibration_ratio=calibration_ratio,
            )

            if not critic_items:
                logger.info(
                    f"  [{critic.name}] inactive: specialty pool below "
                    f"threshold; no specialist DPO pairs selected"
                )
                return []

            # Build lookup dict for O(1) matching instead of O(N*M) scan
            pair_index: dict[tuple[str, str], dict] = {}
            for p in raw_pairs:
                key = (p["sample"].get("question", ""), p["actor_response"])
                pair_index[key] = p

            for item in critic_items:
                key = (item.sample.get("question", ""), item.response)
                p = pair_index.get(key)
                if p is not None:
                    preference_pairs.append({
                        "sample": p["sample"],
                        "chosen": p["chosen"],
                        "rejected": p["rejected"],
                        "actor_response": p.get("actor_response", ""),
                        "metadata": {
                            "target_skill": specialty.value,
                            "assigned_skill": item.skill.value if item.skill else "general",
                            "source_bucket": item.source_bucket,
                            "routing_weight": item.weight,
                            "error_profile": item.profile,
                            "delta": p["delta"],
                            "direction": p["direction"],
                        },
                    })

        logger.info(
            f"  [{critic.name}] {len(preference_pairs)}/{len(raw_pairs)} pairs "
            f"selected for skill '{specialty.value if specialty else 'none'}'"
        )

        # Log assigned_skill distribution to verify routing is effective
        if preference_pairs:
            skill_dist = Counter(
                p["metadata"]["assigned_skill"]
                for p in preference_pairs
            )
            bucket_dist = Counter(
                p["metadata"]["source_bucket"]
                for p in preference_pairs
            )
            selected_unique_pairs = {
                (p["sample"].get("question", ""), p.get("actor_response", ""))
                for p in preference_pairs
            }
            logger.info(
                f"  [{critic.name}] source_bucket distribution: "
                f"{dict(bucket_dist)}"
            )
            logger.info(
                f"  [{critic.name}] assigned_skill distribution: "
                f"{dict(skill_dist)}"
            )
            logger.info(
                f"  [{critic.name}] selected unique_pairs: "
                f"{len(selected_unique_pairs)} / {len(preference_pairs)}"
            )

        if _owns_engine:
            del engine
            _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to generate critic pairs for {critic.name}: {e}")
        if _owns_engine:
            _cleanup_gpu()
        if strict_classification:
            raise

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
    max_tokens: int,
    reward_threshold: float,
    max_samples: int,
    seed: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    engine: Any = None,
    classifications_cache_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
    strict_classification: bool = True,
    max_classification_failure_rate: float = 0.0,
    request_timeout: int | float = 30,
    max_retries: int = 5,
    retry_delay: int | float = 5,
) -> list[dict]:
    """Generate Actor DPO preference pairs using Algorithm 1 from trajectory.py.

    Round-robins across all available critics (one per sample) so the actor
    learns from diverse critic feedback styles, matching the MoE Top-K
    routing at inference.  Each ``generate_trajectories`` call still uses
    only 1 actor + 1 critic LoRA concurrently.

    The actor preference pairs use:
      chosen  = positive (guided-toward-correct actor response)
      rejected = negative (natural / guided-toward-wrong actor response)

    Reasoning-style filtering ensures data-level diversification.
    """
    from src.inference.vllm_server import VLLMInference
    from src.algorithms.deliberation import deliberate_batch
    from src.algorithms.trajectory import _generate_guided_pairs_for_batch, generate_wrong_answer

    model_name = registry.base_model_path or "Qwen/Qwen2.5-7B-Instruct"

    all_agents = [actor] + list(critics)

    n_samples = min(len(dataset), max_samples)
    preference_pairs: list[dict] = []
    _owns_engine = engine is None

    try:
        if _owns_engine:
            enable_lora = any(a.lora_path for a in all_agents)
            max_loras = sum(1 for a in all_agents if a.lora_path) if enable_lora else 0
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

        critics_with_lora = [c for c in critics if c.lora_path]
        critics_pool = critics_with_lora if critics_with_lora else critics

        samples_subset = dataset[:n_samples]

        # Group samples by their round-robin critic for batched deliberation.
        critic_groups: dict[str, list[tuple[int, dict]]] = {}
        for i, sample in enumerate(samples_subset):
            ref_critic = critics_pool[i % len(critics_pool)]
            critic_groups.setdefault(ref_critic.name, []).append((i, sample))

        raw_pairs: list[dict] = []

        for critic_name, indexed_samples in critic_groups.items():
            critic_adapter = adapters[critic_name]
            group_samples = [s for _, s in indexed_samples]

            logger.info(
                f"  [{actor.name}] Batched with critic {critic_name}: "
                f"{len(group_samples)} samples"
            )

            # Batched natural deliberation
            trajectories = deliberate_batch(
                actor_adapter, critic_adapter, group_samples, dataset_name,
                num_rounds=num_rounds, max_tokens=max_tokens, temperature=0.7,
            )

            correct_answers: list[str] = []
            wrong_answers: list[str] = []
            for j, sample in enumerate(group_samples):
                rng = random.Random(seed + indexed_samples[j][0])
                correct_answer = sample.get("answer", "")
                task_type = sample.get("task_type", "math")
                correct_answers.append(correct_answer)
                wrong_answers.append(generate_wrong_answer(
                    correct_answer, sample.get("choices"),
                    task_type=task_type, rng=rng,
                ))

            algo_pairs = _generate_guided_pairs_for_batch(
                actor_adapter, critic_adapter, group_samples, trajectories,
                dataset_name, correct_answers, wrong_answers,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            logger.info(
                f"  [{actor.name}] Guided+MC batch with critic {critic_name}: "
                f"{len(algo_pairs)} Algorithm 1 pairs"
            )

            for pair in algo_pairs:
                raw_pairs.append({
                    "sample": pair["sample"],
                    "chosen": pair["positive"],
                    "rejected": pair["negative"],
                    "delta": pair["delta"],
                    "direction": pair["direction"],
                })

        logger.info(
            f"  [{actor.name}] Algorithm 1 produced {len(raw_pairs)} raw actor pairs"
        )

        # Step 2: Batch classify reasoning styles, then filter
        if raw_pairs:
            # Deduplicate classification calls by (question, response) key.
            # Many raw_pairs share the same response (e.g. same actor output
            # across different rounds), so we classify each unique response
            # exactly once.
            style_cache: dict[tuple[str, str], Optional[ReasoningStyle]] = {}
            classify_keys: list[tuple[str, str, str, str]] = []  # (q, resp, ans, key)

            for p in raw_pairs:
                sample = p["sample"]
                question = sample.get("question", "")
                chosen = p["chosen"]
                correct_answer = sample.get("answer", "")
                key = (question, chosen)
                if key not in style_cache:
                    style_cache[key] = None  # sentinel: pending
                    classify_keys.append((question, chosen, correct_answer, key))

            # Batch classify: each call checks disk cache first, then API.
            # The per-file disk cache in data_classifier.py avoids redundant
            # API calls across runs.  We consolidate here to deduplicate
            # identical (question, response) pairs.
            n_ok = 0
            n_failed = 0
            for question, response, correct_answer, key in classify_keys:
                try:
                    result = classify_reasoning_style(
                        response=response,
                        question=question,
                        correct_answer=correct_answer,
                        use_api=True,
                        cache_dir=classifications_cache_dir,
                        api_key=api_key,
                        api_base=api_base,
                        api_model=api_model,
                        request_timeout=request_timeout,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                    )
                    style_cache[key] = result.style
                    n_ok += 1
                except ClassificationError:
                    n_failed += 1

            logger.info(
                f"  [{actor.name}] Style classification: "
                f"{len(style_cache)} unique responses, "
                f"{n_ok} classified, {n_failed} failures"
            )
            attempted = n_ok + n_failed
            failure_rate = n_failed / attempted if attempted else 0.0
            if strict_classification and failure_rate > max_classification_failure_rate:
                raise ClassificationError(
                    f"reasoning_style classification failure rate {failure_rate:.3f} "
                    f"exceeds threshold {max_classification_failure_rate:.3f} "
                    f"({n_failed}/{attempted})"
                )

            # Filter pairs by actor's reasoning style using cached results
            target_style = actor.reasoning_style
            for p in raw_pairs:
                sample = p["sample"]
                question = sample.get("question", "")
                chosen = p["chosen"]
                key = (question, chosen)

                style = style_cache.get(key)
                if style is not None and style == target_style:
                    preference_pairs.append({
                        "sample": sample,
                        "chosen": chosen,
                        "rejected": p["rejected"],
                        "metadata": {
                            "style": target_style.value,
                            "delta": p["delta"],
                            "direction": p["direction"],
                        },
                    })

        logger.info(
            f"  [{actor.name}] {len(preference_pairs)}/{len(raw_pairs)} pairs "
            f"matched style '{actor.reasoning_style.value}'"
        )

        if _owns_engine:
            del engine
            _cleanup_gpu()

    except Exception as e:
        logger.error(f"Failed to generate actor pairs for {actor.name}: {e}")
        if _owns_engine:
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
    dataset_name: str = "math",
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
        from src.prompts.formatter import format_prompt
        from src.prompts.templates import PromptType

        # Reconstruct full prompts using the same template as generation.
        # Actor pairs  -> SINGLE_SHOT (question + passage + choices)
        # Critic pairs -> DELIBERATION_CRITIC (question + passage + actor_response)
        prompts = []
        for p in preference_pairs:
            sample = p.get("sample", {})
            if agent_type == "critic" and p.get("actor_response"):
                prompt = format_prompt(
                    dataset_name,
                    PromptType.DELIBERATION_CRITIC,
                    sample,
                    actor_response=p["actor_response"],
                )
            else:
                prompt = format_prompt(
                    dataset_name,
                    PromptType.SINGLE_SHOT,
                    sample,
                    include_answer_contract=(agent_type == "actor"),
                )
            prompts.append(prompt)

        # Convert preference_pairs to HuggingFace Dataset
        hf_data = {
            "prompt": prompts,
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
