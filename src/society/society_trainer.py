"""
Society trainer: Multi-Agent alternating training scheduler.

For each iteration:
  Phase A: Fix all Actors -> Train each Critic on a mixture of general,
           specialty, and calibration data routed by multi-dimensional error profiles.
  Phase B: Fix all Critics -> Train each Actor on its reasoning-style subset.

Data routing uses error-profile classification to construct critic-specific
mixture datasets (general/specialty/calibration).  The routing_weight stored
in metadata reflects the mixture sampling probability — it is NOT applied
directly to the DPO loss.  Actual DPO training uses unweighted loss on the
mixture dataset.

Preference pairs are generated with Society-native pairwise dialogues, guided
counterfactual responses, and rollout scoring.  The training path does not
depend on the removed standalone 1 Actor + 1 Critic experiment stack.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.society.agent_registry import (
    ACTOR_STYLE_PROMPTS,
    AgentRegistry,
    AgentConfig,
    AgentRole,
    resolve_critic_skill,
    ReasoningStyle,
)
from src.prompts.critic_prompts import render_critic_judgement
from src.prompts.prompt_builder import (
    build_critic_feedback_prompt,
    build_simple_actor_prompt,
)
from src.society.data_classifier import (
    classify_reasoning_style, ClassificationError,
)
from src.society.diversity_split import DiversitySplit, summarize_critic_training_pairs
from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer
from src.society.pair_generation import (
    build_guided_rollout_pairs,
    run_pairwise_deliberation_batch,
)

logger = logging.getLogger(__name__)


ACTOR_STYLE_TRAINING_GUIDANCE = {
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


def _style_condition_actor_prompt(prompt: str, style: ReasoningStyle) -> str:
    """Prefix an Actor prompt with the requested reasoning style."""

    return (
        "/no_think\n"
        f"You are Actor-{style.value}.\n"
        "Use this reasoning style naturally.\n"
        f"{ACTOR_STYLE_PROMPTS[style]}\n"
        f"{ACTOR_STYLE_TRAINING_GUIDANCE[style]}\n\n"
        f"{prompt}"
    )


def _style_output_format(style: ReasoningStyle) -> str:
    if style in {
        ReasoningStyle.DIRECT,
        ReasoningStyle.EVIDENCE,
        ReasoningStyle.ELIMINATION,
    }:
        return (
            "Reason naturally in the assigned style.\n"
            "At the end, write one final answer sentence:\n"
            "The final result is <answer>."
        )
    raise ValueError(f"Unsupported reasoning style: {style}")


def _style_condition_actor_single_shot_prompt(
    dataset_name: str,
    sample: dict[str, Any],
    style: ReasoningStyle,
) -> str:
    base_prompt = build_simple_actor_prompt(sample, dataset_name, style=style)
    return (
        _style_condition_actor_prompt(base_prompt, style)
        + "\n\n"
        + _style_output_format(style)
    )


class StyleConditionedActorAdapter:
    """Wrap an actor model so every generated Actor prompt is style-conditioned."""

    def __init__(self, model: Any, style: ReasoningStyle):
        self._model = model
        self._style = style

    def generate(self, prompts, max_tokens=256, temperature=0.7, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        styled_prompts = [
            _style_condition_actor_prompt(prompt, self._style)
            for prompt in prompts
        ]
        return self._model.generate(
            styled_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def generate_single(self, prompt, max_tokens=256, temperature=0.7, **kwargs):
        results = self.generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return results[0] if results else ""


def _cleanup_gpu():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _release_vllm_engine(engine: Any, adapters: Optional[dict[str, Any]] = None) -> None:
    """Explicitly release a vLLM engine and adapter references."""
    if engine is not None:
        try:
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()
        except Exception as e:
            logger.warning(f"vLLM engine cleanup failed: {e}")

    if adapters:
        for adapter in adapters.values():
            if hasattr(adapter, "_engine"):
                adapter._engine = None
        adapters.clear()

    _cleanup_gpu()


def _render_structured_critic_chosen(pair: dict, profile: dict) -> str:
    primary = str(profile.get("primary", "verification")).strip().lower()
    if primary not in {"computation", "reasoning", "knowledge", "grounding", "verification"}:
        primary = "verification"
    confidence = profile.get("confidence", 0.8)
    try:
        confidence = max(0.1, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.8
    evidence = str(profile.get("evidence", "")).strip()
    if not evidence:
        evidence = "The actor response is inconsistent with the verified answer and needs correction."
    del primary
    return render_critic_judgement(
        answer_correct="no",
        suggested_answer=str(pair.get("correct_answer") or "unknown"),
        confidence=confidence,
        critique=evidence,
    )


def _render_structured_critic_rejected(pair: dict, profile: dict, target_skill: str) -> str:
    actor_answer = str(pair.get("actor_answer") or "unknown")
    primary = str(profile.get("primary", target_skill)).strip().lower()
    if primary not in {"computation", "reasoning", "knowledge", "grounding", "verification"}:
        primary = target_skill
    del primary
    return render_critic_judgement(
        answer_correct="yes",
        suggested_answer=actor_answer,
        confidence=0.80,
        critique="The actor answer is acceptable.",
    )


def _create_phase_engine(
    model_name: str,
    agents: list[AgentConfig],
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_concurrent_loras: int | None = None,
    tensor_parallel_size: int = 1,
):
    """Create a shared VLLMInference engine for all agents in a training phase.

    The engine is configured to accommodate LoRA adapters for all agents
    that have lora_path set.

    Args:
        max_concurrent_loras: Override for max_loras.  When *None*, defaults to
            the number of agents that have a LoRA path (one slot per agent).
            Set this to a smaller number (e.g. 2) when only a subset of agents
            are active concurrently, to reduce GPU memory reservation.
        tensor_parallel_size: Number of GPUs for tensor parallelism.  When > 1,
            uses GPUs ``[device, device + tensor_parallel_size)``.
    """
    from src.inference.vllm_server import VLLMInference

    enable_lora = any(a.lora_path for a in agents)

    if max_concurrent_loras is not None:
        max_loras = max_concurrent_loras
    else:
        max_loras = sum(1 for a in agents if a.lora_path) if enable_lora else 0

    if tensor_parallel_size > 1:
        cuda_device = list(range(device, device + tensor_parallel_size))
    else:
        cuda_device = device

    return VLLMInference(
        model_name,
        cuda_device=cuda_device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=enable_lora or None,
        max_loras=max(max_loras, 1),
        max_lora_rank=256,
    )


def _save_pairs_json(pairs: list[dict], path: str) -> None:
    """Save preference pairs to a JSON file for checkpoint/resume."""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)


def _load_pairs_json(path: str) -> list[dict]:
    """Load preference pairs from a JSON cache file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    tensor_parallel_size: int = 1,
    classifications_cache_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
    strict_classification: bool = True,
    max_classification_failure_rate: float = 0.0,
    max_classification_workers: int = 4,
    min_style_confidence: float = 0.65,
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
        trajectory_max_tokens: Max tokens for Society pairwise rollout generation.
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

        # ---- Phase A: Train all Critics (fix Actors) ----
        logger.info(
            f"Phase A: Training {len(critics)} Critics "
            f"(Actors frozen, error-profile mixture routing)"
        )

        # Shared engine for Phase A: reuse across critics to avoid
        # repeated model loading (~1 min each).  The engine is destroyed
        # before DPO training (which loads its own model).
        all_phase_a_agents = list(actors) + list(critics)
        phase_a_engine = None

        phase_a_done = set(phase_done.get("phase_A", []))
        critic_pairs_cache: dict[str, list[dict]] = {}

        # --- Pass 1: Generate all preference pairs with shared engine ---
        for critic in critics:
            if critic.name in phase_a_done:
                logger.info(
                    f"  Skipping Critic {critic.display_name} (already completed)"
                )
                continue

            critic_iter_dir = f"{output_base_dir}/critics/{critic.name}/iter_{iteration}"
            Path(critic_iter_dir).mkdir(parents=True, exist_ok=True)
            pairs_file = os.path.join(critic_iter_dir, "preference_pairs.json")

            # Check cached pairs from a previous interrupted run
            if os.path.exists(pairs_file):
                logger.info(
                    f"  Loading cached pairs for {critic.display_name} "
                    f"from {pairs_file}"
                )
                critic_pairs_cache[critic.name] = _load_pairs_json(pairs_file)
                continue

            # Create shared engine lazily (only when at least one critic
            # actually needs pair generation)
            if phase_a_engine is None:
                logger.info("  Creating shared vLLM engine for Phase A")
                phase_a_engine = _create_phase_engine(
                    registry.base_model_path or "Qwen/Qwen3-14B",
                    all_phase_a_agents,
                    device=device,
                    dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    tensor_parallel_size=tensor_parallel_size,
                )

            logger.info(
                f"  Generating pairs for Critic: {critic.display_name} "
                f"(specialty: {critic.error_specialty.value})"
            )

            preference_pairs = _generate_critic_pairs_pairwise(
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
                engine=phase_a_engine,
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

            # Cache pairs for potential resume
            if preference_pairs:
                _save_pairs_json(preference_pairs, pairs_file)
            critic_pairs_cache[critic.name] = preference_pairs

        # Release shared engine before DPO training
        if phase_a_engine is not None:
            _release_vllm_engine(phase_a_engine)
            phase_a_engine = None
        _cleanup_gpu()

        # --- Pass 2: Run DPO training for each critic ---
        for critic in critics:
            if critic.name in phase_a_done:
                continue

            preference_pairs = critic_pairs_cache.get(critic.name, [])

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

            critic_iter_dir = f"{output_base_dir}/critics/{critic.name}/iter_{iteration}"

            # Run DPO training
            logger.info(f"  Training with {len(preference_pairs)} preference pairs")
            checkpoint_path = _run_dpo_training(
                model_name=registry.base_model_path or "Qwen/Qwen3-14B",
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

            # Per-critic checkpoint
            phase_a_done.add(critic.name)
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

        # Cleanup Phase A engine (may already be None)
        if phase_a_engine is not None:
            _release_vllm_engine(phase_a_engine)
            phase_a_engine = None
        _cleanup_gpu()

        _sync_lora_paths(critics, critic_paths)

        # ---- Phase B: Train all Actors (fix Critics) ----
        logger.info(f"Phase B: Training {len(actors)} Actors (Critics frozen)")

        # Shared engine for Phase B: reuse across actors to avoid
        # repeated model loading.
        all_phase_b_agents = list(actors) + list(critics)
        phase_b_engine = None

        phase_b_done = set(phase_done.get("phase_B", []))
        actor_pairs_cache: dict[str, list[dict]] = {}

        # --- Pass 1: Generate all preference pairs with shared engine ---
        for actor in actors:
            if actor.name in phase_b_done:
                logger.info(
                    f"  Skipping Actor {actor.display_name} (already completed)"
                )
                continue

            actor_iter_dir = f"{output_base_dir}/actors/{actor.name}/iter_{iteration}"
            Path(actor_iter_dir).mkdir(parents=True, exist_ok=True)
            pairs_file = os.path.join(actor_iter_dir, "preference_pairs.json")

            # Check cached pairs from a previous interrupted run
            if os.path.exists(pairs_file):
                logger.info(
                    f"  Loading cached pairs for {actor.display_name} "
                    f"from {pairs_file}"
                )
                actor_pairs_cache[actor.name] = _load_pairs_json(pairs_file)
                continue

            # Create shared engine lazily
            if phase_b_engine is None:
                logger.info("  Creating shared vLLM engine for Phase B")
                phase_b_engine = _create_phase_engine(
                    registry.base_model_path or "Qwen/Qwen3-14B",
                    all_phase_b_agents,
                    device=device,
                    dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    tensor_parallel_size=tensor_parallel_size,
                )

            logger.info(
                f"  Generating pairs for Actor: {actor.display_name} "
                f"(style: {actor.reasoning_style.value})"
            )

            preference_pairs = _generate_actor_pairs_pairwise(
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
                engine=phase_b_engine,
                classifications_cache_dir=classifications_cache_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                strict_classification=strict_classification,
                max_classification_failure_rate=max_classification_failure_rate,
                max_classification_workers=max_classification_workers,
                min_style_confidence=min_style_confidence,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

            # Cache pairs for potential resume
            if preference_pairs:
                _save_pairs_json(preference_pairs, pairs_file)
            actor_pairs_cache[actor.name] = preference_pairs

        # Release shared engine before DPO training
        if phase_b_engine is not None:
            _release_vllm_engine(phase_b_engine)
            phase_b_engine = None
        _cleanup_gpu()

        # --- Pass 2: Run DPO training for each actor ---
        for actor in actors:
            if actor.name in phase_b_done:
                continue

            preference_pairs = actor_pairs_cache.get(actor.name, [])

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
                        "phase_A": sorted(phase_a_done),
                        "phase_B": sorted(phase_b_done),
                    },
                })
                continue

            actor_iter_dir = f"{output_base_dir}/actors/{actor.name}/iter_{iteration}"

            # Run DPO training
            logger.info(f"  Training with {len(preference_pairs)} preference pairs")
            checkpoint_path = _run_dpo_training(
                model_name=registry.base_model_path or "Qwen/Qwen3-14B",
                preference_pairs=preference_pairs,
                output_dir=actor_iter_dir,
                agent_type="actor",
                agent_name=actor.name,
                dataset_name=dataset_name,
                actor_style=actor.reasoning_style,
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
            phase_b_done.add(actor.name)
            _save_checkpoint(ckpt_file, {
                "iteration": iteration,
                "actor_paths": actor_paths,
                "critic_paths": critic_paths,
                "metrics": metrics,
                "phase_done": {
                    **phase_done,
                    "phase_A": sorted(phase_a_done),
                    "phase_B": [
                        n for n, p in actor_paths.items()
                        if metrics.get(f"actor_{n}_iter{iteration}", {}).get("status") == "completed"
                    ],
                },
            })

        # Cleanup Phase B engine (may already be None)
        if phase_b_engine is not None:
            _release_vllm_engine(phase_b_engine)
            phase_b_engine = None
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
# LoRA Model Adapter — bridges shared VLLMInference + LoRA to the simple
# generation interface used by Society pair generation.
# ============================================================

class LoRAModelAdapter:
    """Wraps a shared VLLMInference engine with an optional LoRA adapter.

    Provides ``generate()`` and ``generate_single()`` while routing generation
    calls through the single shared engine with per-agent LoRA requests.
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
) -> dict[str, LoRAModelAdapter]:
    """Create LoRAModelAdapter for each agent.

    Agents with ``lora_path`` must successfully load that adapter.  Falling
    back to the base model would invalidate multi-agent experiments because
    training logs could claim an agent participated while its specialized
    weights were never used.
    """
    from src.society.multi_deliberation import LoRAError, _load_lora_adapter

    adapters: dict[str, LoRAModelAdapter] = {}
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
        adapters[agent.name] = LoRAModelAdapter(engine, lora_req)
    return adapters


# ============================================================
# Phase A: Critic preference pairs via Society pairwise rollouts
# ============================================================

def _generate_critic_pairs_pairwise(
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
    """Generate Critic DPO pairs from Society pairwise guided rollouts.

    Round-robins across all available actors (one per sample) so the critic
    learns to give feedback for diverse actor styles, matching the multi-actor
    inference setting. Each pairwise rollout uses one Actor adapter and one
    Critic adapter concurrently.

    Error-profile routing via DiversitySplit constructs a critic-specific mixture
    dataset (general/specialty/calibration).  The routing_weight stored in
    metadata reflects mixture sampling probability, NOT a direct DPO loss
    weighting — actual DPO training uses unweighted loss on the mixture.
    """
    from src.inference.vllm_server import VLLMInference

    model_name = registry.base_model_path or "Qwen/Qwen3-14B"
    specialty = critic.error_specialty

    all_agents = list(actors) + [critic]

    n_samples = min(len(dataset), max_samples)
    preference_pairs: list[dict] = []
    _owns_engine = engine is None
    adapters: dict[str, Any] = {}

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

            trajectories = run_pairwise_deliberation_batch(
                actor_adapter, critic_adapter, group_samples, dataset_name,
                num_rounds=num_rounds, max_tokens=max_tokens, temperature=0.7,
            )

            correct_answers: list[str] = []
            wrong_answers: list[str] = []
            for j, sample in enumerate(group_samples):
                import random
                from src.data.preprocessor import generate_wrong_answer
                rng = random.Random(seed + indexed_samples[j][0])
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                correct_answers.append(correct_answer)
                wrong_answers.append(generate_wrong_answer(
                    correct_answer, sample.get("choices"),
                    task_type=task_type, rng=rng,
                ))

            rollout_pairs = build_guided_rollout_pairs(
                actor_adapter, critic_adapter, group_samples, trajectories,
                dataset_name, correct_answers, wrong_answers,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            logger.info(
                f"  [{critic.name}] Guided rollout batch with actor {actor_name}: "
                f"{len(rollout_pairs)} candidate pairs"
            )

            for pair in rollout_pairs:
                sample = pair["sample"]
                actor_candidate = pair["actor_candidate"]
                critic_candidate = pair["critic_candidate"]
                rejected_actor_response = actor_candidate["rejected"]
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                rejected_answer = extract_answer(rejected_actor_response, task_type)

                if task_type == "math":
                    is_wrong = not math_answers_equal(
                        rejected_answer or "", correct_answer,
                    )
                else:
                    is_wrong = normalize_answer(
                        rejected_answer or "", task_type,
                    ) != normalize_answer(correct_answer, task_type)

                if is_wrong:
                    raw_pairs.append({
                        "sample": sample,
                        "critic_pair": critic_candidate,
                        "actor_pair": actor_candidate,
                        "actor_response": rejected_actor_response,
                        "actor_answer": rejected_answer,
                        "correct_answer": correct_answer,
                        "task_type": task_type,
                        "delta": pair["comparison"]["delta"],
                        "comparison_mode": pair["comparison"]["mode"],
                        "rollout_scores": pair["rollout_scores"],
                    })

        logger.info(
            f"  [{critic.name}] Pairwise rollouts produced {len(raw_pairs)} raw critic pairs"
        )

        # Routing below is API-bound and can take several minutes. Release the
        # local vLLM engine first so the process is not holding an idle GPU.
        if _owns_engine and engine is not None:
            _release_vllm_engine(engine, adapters)
            engine = None

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
                    profile = item.profile if isinstance(item.profile, dict) else {}
                    chosen = _render_structured_critic_chosen(p, profile)
                    rejected = _render_structured_critic_rejected(p, profile, specialty.value)
                    preference_pairs.append({
                        "sample": p["sample"],
                        "chosen": chosen,
                        "rejected": rejected,
                        "actor_response": p.get("actor_response", ""),
                        "metadata": {
                            "target_skill": specialty.value,
                            "assigned_skill": item.skill.value if item.skill else "general",
                            "source_bucket": item.source_bucket,
                            "routing_weight": item.weight,
                            "error_profile": profile,
                            "delta": p["delta"],
                            "comparison_mode": p["comparison_mode"],
                            "rollout_scores": p.get("rollout_scores", {}),
                            "actor_answer": p.get("actor_answer", ""),
                            "correct_answer": p.get("correct_answer", ""),
                            "structured_judgement": True,
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

        if _owns_engine and engine is not None:
            _release_vllm_engine(engine, adapters)
            engine = None

    except Exception as e:
        logger.error(f"Failed to generate critic pairs for {critic.name}: {e}")
        if _owns_engine and engine is not None:
            _release_vllm_engine(engine, adapters)
            engine = None
        if strict_classification:
            raise

    return preference_pairs


# ============================================================
# Phase B: Actor preference pairs via Society pairwise rollouts
# ============================================================

def _generate_actor_pairs_pairwise(
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
    max_classification_workers: int = 4,
    min_style_confidence: float = 0.65,
    request_timeout: int | float = 30,
    max_retries: int = 5,
    retry_delay: int | float = 5,
) -> list[dict]:
    """Generate Actor DPO pairs from Society pairwise guided rollouts.

    Round-robins across all available critics (one per sample) so the actor
    learns from diverse critic feedback styles, matching the MoE Top-K
    routing at inference. Each pairwise rollout uses one Actor adapter and one
    Critic adapter concurrently.

    Reasoning-style filtering ensures data-level diversification.
    """
    from src.inference.vllm_server import VLLMInference

    model_name = registry.base_model_path or "Qwen/Qwen3-14B"

    all_agents = [actor] + list(critics)

    n_samples = min(len(dataset), max_samples)
    preference_pairs: list[dict] = []
    _owns_engine = engine is None
    adapters: dict[str, Any] = {}

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
        if actor.reasoning_style is None:
            raise ValueError(f"Actor '{actor.name}' has no reasoning_style set")

        actor_adapter = StyleConditionedActorAdapter(
            adapters[actor.name],
            actor.reasoning_style,
        )

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

            trajectories = run_pairwise_deliberation_batch(
                actor_adapter, critic_adapter, group_samples, dataset_name,
                num_rounds=num_rounds, max_tokens=max_tokens, temperature=0.7,
            )

            correct_answers: list[str] = []
            wrong_answers: list[str] = []
            for j, sample in enumerate(group_samples):
                import random
                from src.data.preprocessor import generate_wrong_answer
                rng = random.Random(seed + indexed_samples[j][0])
                correct_answer = sample.get("answer", "")
                task_type = sample.get("task_type", "math")
                correct_answers.append(correct_answer)
                wrong_answers.append(generate_wrong_answer(
                    correct_answer, sample.get("choices"),
                    task_type=task_type, rng=rng,
                ))

            rollout_pairs = build_guided_rollout_pairs(
                actor_adapter, critic_adapter, group_samples, trajectories,
                dataset_name, correct_answers, wrong_answers,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            logger.info(
                f"  [{actor.name}] Guided rollout batch with critic {critic_name}: "
                f"{len(rollout_pairs)} candidate pairs"
            )

            for pair in rollout_pairs:
                raw_pairs.append({
                    "sample": pair["sample"],
                    "actor_pair": pair["actor_candidate"],
                    "critic_pair": pair["critic_candidate"],
                    "delta": pair["comparison"]["delta"],
                    "comparison_mode": pair["comparison"]["mode"],
                    "rollout_scores": pair["rollout_scores"],
                })

        logger.info(
            f"  [{actor.name}] Pairwise rollouts produced {len(raw_pairs)} raw actor pairs"
        )

        # Style filtering is API-bound; release vLLM before classification to
        # avoid holding an idle GPU during potentially slow external calls.
        if _owns_engine and engine is not None:
            _release_vllm_engine(engine, adapters)
            engine = None

        # Step 2: Batch classify reasoning styles, then filter
        if raw_pairs:
            # Deduplicate classification calls by (question, response) key.
            # Many raw_pairs share the same response (e.g. same actor output
            # across different rounds), so we classify each unique response
            # exactly once.
            style_cache: dict[tuple[str, str], Optional[dict[str, Any]]] = {}
            classify_keys: list[tuple[str, str, str, tuple[str, str]]] = []

            for p in raw_pairs:
                sample = p["sample"]
                question = sample.get("question", "")
                chosen = p["actor_pair"]["chosen"]
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
            def classify_one(payload: tuple[str, str, str, tuple[str, str]]):
                question, response, correct_answer, key = payload
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
                    return key, {
                        "style": result.style,
                        "format_status": result.format_status,
                        "confidence": result.confidence,
                    }, None
                except ClassificationError:
                    return key, None, "classification_error"

            if max_classification_workers == 1:
                classify_results = [classify_one(payload) for payload in classify_keys]
            else:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max(1, max_classification_workers)) as executor:
                    classify_results = list(executor.map(classify_one, classify_keys))

            for key, result, error in classify_results:
                if error is None:
                    style_cache[key] = result
                    n_ok += 1
                else:
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
            prompt_style = target_style.value
            min_conf = float(min_style_confidence)
            for p in raw_pairs:
                sample = p["sample"]
                question = sample.get("question", "")
                chosen = p["actor_pair"]["chosen"]
                key = (question, chosen)
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                chosen_answer = extract_answer(chosen, task_type)
                if task_type == "math":
                    is_correct = math_answers_equal(
                        chosen_answer or "",
                        correct_answer,
                    )
                else:
                    is_correct = (
                        normalize_answer(chosen_answer or "", task_type)
                        == normalize_answer(correct_answer, task_type)
                    )

                result = style_cache.get(key)
                classified_style = result.get("style") if result else None
                format_status = result.get("format_status") if result else None
                style_confidence = float(result.get("confidence", 0.0)) if result else 0.0
                accepted_for_actor = (
                    is_correct
                    and prompt_style == target_style.value
                    and classified_style == target_style
                    and format_status == "valid"
                    and style_confidence >= min_conf
                )
                if accepted_for_actor:
                    preference_pairs.append({
                        "sample": sample,
                        "chosen": chosen,
                        "rejected": p["actor_pair"]["rejected"],
                        "metadata": {
                            "style": target_style.value,
                            "prompted_style": prompt_style,
                            "classified_style": target_style.value,
                            "style_confidence": style_confidence,
                            "format_status": format_status,
                            "is_correct": True,
                            "chosen_answer": chosen_answer,
                            "accepted_for_actor": True,
                            "delta": p["delta"],
                            "comparison_mode": p["comparison_mode"],
                            "rollout_scores": p.get("rollout_scores", {}),
                        },
                    })

        logger.info(
            f"  [{actor.name}] {len(preference_pairs)}/{len(raw_pairs)} pairs "
            f"matched style '{actor.reasoning_style.value}'"
        )

        if _owns_engine and engine is not None:
            _release_vllm_engine(engine, adapters)
            engine = None

    except Exception as e:
        logger.error(f"Failed to generate actor pairs for {actor.name}: {e}")
        if _owns_engine and engine is not None:
            _release_vllm_engine(engine, adapters)
            engine = None
        if strict_classification:
            raise

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
    actor_style: ReasoningStyle | None = None,
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

        prompts = []
        for p in preference_pairs:
            sample = p.get("sample", {})
            if agent_type == "critic" and p.get("actor_response"):
                target_skill = p.get("metadata", {}).get("target_skill", "")
                error_specialty = None
                if target_skill:
                    try:
                        error_specialty = resolve_critic_skill(target_skill)
                    except ValueError:
                        error_specialty = None
                critic_cfg = AgentConfig(
                    name=f"critic_{target_skill or 'general'}",
                    role=AgentRole.CRITIC,
                    model_path=model_name,
                    error_specialty=error_specialty,
                )
                prompt = build_critic_feedback_prompt(
                    critic_cfg,
                    sample,
                    dataset_name,
                    p["actor_response"],
                )
            elif agent_type == "actor":
                style = actor_style
                if style is None:
                    style_label = (
                        p.get("metadata", {}).get("prompted_style")
                        or p.get("metadata", {}).get("style")
                    )
                    if style_label:
                        style = ReasoningStyle(style_label)
                if style is not None:
                    prompt = _style_condition_actor_single_shot_prompt(
                        dataset_name,
                        sample,
                        style,
                    )
                else:
                    prompt = build_simple_actor_prompt(
                        sample,
                        dataset_name,
                    )
            else:
                prompt = build_simple_actor_prompt(
                    sample,
                    dataset_name,
                )
            prompts.append(prompt)

        # Convert preference_pairs to HuggingFace Dataset
        hf_data = {
            "prompt": prompts,
            "chosen": [p["chosen"] for p in preference_pairs],
            "rejected": [p["rejected"] for p in preference_pairs],
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
