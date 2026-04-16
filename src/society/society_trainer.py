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

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.society.agent_registry import AgentRegistry, AgentRole, ReasoningStyle, ErrorType
from src.society.diversity_split import DiversitySplit

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

    Returns:
        SocietyTrainingResult with checkpoint paths and metrics.
    """
    actors = registry.list_actors()
    critics = registry.list_critics()

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

    # Initialize diversity splitter
    diversity_split = DiversitySplit(balance=True, seed=seed)

    for iteration in range(start_iteration, num_iterations):
        logger.info(f"=== Society Training Iteration {iteration + 1}/{num_iterations} ===")

        # ---- Phase A: Train all Critics (fix Actors) ----
        logger.info(f"Phase A: Training {len(critics)} Critics (Actors frozen)")

        # Generate trajectories with current Actors
        # Then classify error types for trajectory data
        for critic in critics:
            critic_iter_dir = f"{output_base_dir}/critics/{critic.name}/iter_{iteration}"
            Path(critic_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Critic: {critic.display_name} "
                f"(specialty: {critic.error_specialty.value})"
            )

            # TODO: In production, this would:
            # 1. Load an Actor + base model → generate deliberation trajectories
            # 2. Classify error types on wrong responses
            # 3. Filter trajectories matching this Critic's error_specialty
            # 4. Build DPO pairs (chosen=Critic correct guidance, rejected=Critic unhelpful)
            # 5. Cleanup vLLM → run DPO training
            # 6. Merge LoRA → save

            critic_paths[critic.name] = critic_iter_dir
            metrics[f"critic_{critic.name}_iter{iteration}"] = {"status": "pending"}

        # ---- Phase B: Train all Actors (fix Critics) ----
        logger.info(f"Phase B: Training {len(actors)} Actors (Critics frozen)")

        for actor in actors:
            actor_iter_dir = f"{output_base_dir}/actors/{actor.name}/iter_{iteration}"
            Path(actor_iter_dir).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  Training Actor: {actor.display_name} "
                f"(style: {actor.reasoning_style.value})"
            )

            # TODO: In production, this would:
            # 1. Load base model → generate bootstrap responses
            # 2. Classify reasoning styles on responses
            # 3. Filter responses matching this Actor's reasoning_style
            # 4. Build DPO pairs (chosen=correct+style-matching, rejected=wrong/different-style)
            # 5. Cleanup vLLM → run DPO training
            # 6. Merge LoRA → save

            actor_paths[actor.name] = actor_iter_dir
            metrics[f"actor_{actor.name}_iter{iteration}"] = {"status": "pending"}

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
    tmp.rename(path)
    logger.info(f"Checkpoint saved: {path}")
