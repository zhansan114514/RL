"""
Inference pipeline for the Diverse Actor-Critic Society.

Supports multiple voting strategies:
- majority_vote: All Actors vote, majority wins
- best_actor: Use highest-confidence Actor's answer
- weighted: Weight Actor answers by Critic confidence

Also supports ablation configurations (A1-A5 from experiment plan).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from src.algorithms.reward import extract_answer
from src.society.agent_registry import AgentRegistry, AgentConfig
from src.society.multi_deliberation import (
    multi_agent_deliberate_single_gpu,
    MultiDeliberationResult,
)
from src.society.router import CriticRouter

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of society inference."""
    final_answer: str
    consensus_confidence: float
    actor_answers: dict[str, Optional[str]]  # actor_name -> answer
    voting_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Main inference function
# ============================================================

def society_inference(
    registry: AgentRegistry,
    sample: dict,
    dataset_name: str,
    inference_engine: Any,  # VLLMInference
    num_actors: Optional[int] = None,  # None = use all
    num_critics: Optional[int] = None,  # None = use all
    num_rounds: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
    voting_strategy: str = "majority_vote",
    router_top_k: int = 2,
    router_uniform: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> InferenceResult:
    """
    Run society inference with configurable voting strategies.

    Supports ablation experiments:
    - A1: num_actors=1, num_critics=1 (baseline ACC-Collab)
    - A2: num_actors=3, num_critics=1 (Actor diversity only)
    - A3: num_actors=1, num_critics=4, router_top_k=2 (Critic specialization only)
    - A4: num_actors=3, num_critics=4, router_top_k=4, router_uniform=True (no routing)
    - A5: num_actors=3, num_critics=4, router_top_k=2, router_uniform=False (full system)

    Args:
        registry: AgentRegistry with all agents.
        sample: Standardized sample dict.
        dataset_name: Dataset name for prompt templates.
        inference_engine: VLLMInference instance.
        num_actors: Number of Actors to use (None = all).
        num_critics: Number of Critics to use (None = all).
        num_rounds: Deliberation rounds.
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        voting_strategy: "majority_vote", "best_actor", or "weighted".
        router_top_k: Top-K Critics for router (use len(critics) for uniform).
        router_uniform: If True, use equal weights instead of softmax confidence.
        checkpoint_dir: Optional crash recovery directory.

    Returns:
        InferenceResult with final answer and metadata.
    """
    all_actors = registry.list_actors()
    all_critics = registry.list_critics()

    # Select subset of agents
    actors = all_actors[:num_actors] if num_actors else all_actors
    critics = all_critics[:num_critics] if num_critics else all_critics

    logger.info(
        f"Society inference: {len(actors)} actors, {len(critics)} critics, "
        f"strategy={voting_strategy}, router_top_k={router_top_k}, "
        f"uniform={router_uniform}"
    )

    # Run multi-agent deliberation
    router = CriticRouter(top_k=router_top_k, uniform_weights=router_uniform)
    frame = inspect.currentframe()
    uniform_weights = frame.f_locals.get('router_uniform', False) if frame else False

    router = CriticRouter(top_k=router_top_k, uniform_weights=uniform_weights)
    result = multi_agent_deliberate_single_gpu(
        inference_engine=inference_engine,
        actors=actors,
        critics=critics,
        sample=sample,
        dataset_name=dataset_name,
        num_rounds=num_rounds,
        max_tokens=max_tokens,
        temperature=temperature,
        router=router,
        checkpoint_dir=checkpoint_dir,
    )

    # Apply voting strategy
    task_type = sample.get("task_type", "yes_no")
    final_answer, confidence = _apply_voting(
        result, voting_strategy, task_type
    )

    return InferenceResult(
        final_answer=final_answer or "",
        consensus_confidence=confidence,
        actor_answers=result.final_answers,
        voting_strategy=voting_strategy,
        metadata={
            "num_actors": len(actors),
            "num_critics": len(critics),
            "router_top_k": router_top_k,
            "num_rounds": num_rounds,
        },
    )


# ============================================================
# Voting strategies
# ============================================================

def _apply_voting(
    result: MultiDeliberationResult,
    strategy: str,
    task_type: str,
) -> tuple[Optional[str], float]:
    """Apply voting strategy to get final answer."""
    answers = {
        name: ans for name, ans in result.final_answers.items()
        if ans is not None
    }

    if not answers:
        return None, 0.0

    if strategy == "majority_vote":
        return _majority_vote(answers)
    elif strategy == "best_actor":
        return _best_actor(answers, result)
    elif strategy == "weighted":
        return _weighted_vote(answers, result)
    else:
        logger.warning(f"Unknown strategy {strategy}, using majority_vote")
        return _majority_vote(answers)


def _majority_vote(
    answers: dict[str, str],
) -> tuple[str, float]:
    """Majority voting across Actor answers."""
    if not answers:
        return "", 0.0
    counter = Counter(answers.values())
    best, count = counter.most_common(1)[0]
    confidence = count / len(answers)
    return best, confidence


def _best_actor(
    answers: dict[str, str],
    result: MultiDeliberationResult,
) -> tuple[str, float]:
    """Select answer from highest-confidence Actor."""
    if not answers:
        return "", 0.0
    # Use the Actor whose answer matches consensus
    if result.consensus_answer and result.consensus_answer in answers.values():
        return result.consensus_answer, result.consensus_confidence
    # Fallback to first actor with reduced confidence
    first_answer = next(iter(answers.values()))
    return first_answer, 0.5


def _weighted_vote(
    answers: dict[str, str],
    result: MultiDeliberationResult,
) -> tuple[str, float]:
    """Weighted voting based on Critic confidence from deliberation rounds.

    Uses the Critic confidence scores from the routing step to weight
    Actor answers. Actors whose answers received higher Critic confidence
    get more weight.
    """
    if not answers:
        return "", 0.0

    # Collect confidence scores from the last deliberation round
    actor_confidences: dict[str, float] = {}
    if result.rounds:
        last_round = result.rounds[-1]
        for actor_name, routed in last_round.routed_feedbacks.items():
            # Average confidence of selected critics for this actor
            critic_confs = [fb.confidence for fb in routed.raw_feedbacks
                           if fb.critic_name in routed.selected_critics]
            if critic_confs:
                actor_confidences[actor_name] = sum(critic_confs) / len(critic_confs)
            else:
                actor_confidences[actor_name] = 0.5

    # Weight answers by actor confidence
    answer_weights: dict[str, float] = {}
    for actor_name, answer in answers.items():
        conf = actor_confidences.get(actor_name, 0.5)
        answer_weights[answer] = answer_weights.get(answer, 0.0) + conf

    if not answer_weights:
        return _majority_vote(answers)

    total_weight = sum(answer_weights.values())
    best_answer = max(answer_weights, key=answer_weights.get)
    confidence = answer_weights[best_answer] / total_weight if total_weight > 0 else 0.0
    return best_answer, confidence


# ============================================================
# Ablation experiment configurations
# ============================================================

ABLATION_CONFIGS = {
    "A1": {"num_actors": 1, "num_critics": 1, "router_top_k": 1, "router_uniform": False,
            "description": "1 Actor + 1 Critic (baseline ACC-Collab)"},
    "A2": {"num_actors": 3, "num_critics": 1, "router_top_k": 1, "router_uniform": False,
            "description": "3 Actors + 1 Critic (Actor diversity only)"},
    "A3": {"num_actors": 1, "num_critics": 4, "router_top_k": 2, "router_uniform": False,
            "description": "1 Actor + 4 Critics + Router (Critic specialization only)"},
    "A4": {"num_actors": 3, "num_critics": 4, "router_top_k": 4, "router_uniform": True,
            "description": "3 Actors + 4 Critics, uniform weights (no routing)"},
    "A5": {"num_actors": 3, "num_critics": 4, "router_top_k": 2, "router_uniform": False,
            "description": "3 Actors + 4 Critics + Router (full system)"},
}


def run_ablation(
    registry: AgentRegistry,
    samples: list[dict],
    dataset_name: str,
    inference_engine: Any,
    configs: Optional[list[str]] = None,
    num_rounds: int = 5,
    checkpoint_dir: Optional[str] = None,
) -> dict[str, list[InferenceResult]]:
    """
    Run ablation experiments A1-A5.

    Args:
        registry: AgentRegistry.
        samples: Test samples.
        dataset_name: Dataset name.
        inference_engine: VLLMInference instance.
        configs: List of ablation config names (e.g., ["A1", "A5"]).
        num_rounds: Deliberation rounds.
        checkpoint_dir: Optional crash recovery.

    Returns:
        Dict mapping config name to list of InferenceResults.
    """
    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())

    results = {}
    for config_name in configs:
        config = ABLATION_CONFIGS[config_name]
        logger.info(f"Running ablation {config_name}: {config['description']}")

        config_results = []
        for i, sample in enumerate(samples):
            result = society_inference(
                registry=registry,
                sample=sample,
                dataset_name=dataset_name,
                inference_engine=inference_engine,
                num_actors=config["num_actors"],
                num_critics=config["num_critics"],
                num_rounds=num_rounds,
                router_top_k=config["router_top_k"],
                router_uniform=config.get("router_uniform", False),
                voting_strategy="majority_vote",
                checkpoint_dir=checkpoint_dir,
            )
            config_results.append(result)

        results[config_name] = config_results
        logger.info(f"  {config_name}: {len(config_results)} samples processed")

    return results
