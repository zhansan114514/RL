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

def _select_agents(
    agents: list,
    count: Optional[int],
    offset: int = 0,
) -> list:
    """Select a subset of agents with deterministic round-robin offset.

    Instead of always picking the lexicographically first agents,
    rotate the starting position by *offset* so that different ablation
    configs (A1, A3, ...) exercise different agents.  This makes the
    single-agent baselines representative rather than biased toward one
    particular name.

    Args:
        agents: Sorted list of agent configs.
        count: Number to select (None = all).
        offset: Round-robin start index (modulo len(agents)).

    Returns:
        Selected subset, preserving the sorted order.
    """
    if count is None or count >= len(agents):
        return list(agents)
    start = offset % len(agents)
    # Build a rotated view then take the first *count* items
    rotated = agents[start:] + agents[:start]
    return rotated[:count]


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
    ablation_label: Optional[str] = None,
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

    # Sort for determinism, then apply round-robin offset so different
    # ablation configs (A1, A3, ...) select different agents instead of
    # always picking the lexicographically first one.
    sorted_actors = sorted(all_actors, key=lambda a: a.name)
    sorted_critics = sorted(all_critics, key=lambda c: c.name)

    # Derive stable offsets from the ablation label so each config
    # exercises a different subset of agents.
    # Use hashlib (deterministic) instead of Python's hash() which is
    # randomized via PYTHONHASHSEED and non-reproducible across runs.
    import hashlib
    actor_offset = int(hashlib.md5((ablation_label or "").encode()).hexdigest(), 16) % max(len(sorted_actors), 1)
    critic_offset = int(hashlib.md5((ablation_label or "").encode()).hexdigest(), 16) % max(len(sorted_critics), 1)

    actors = _select_agents(sorted_actors, num_actors, actor_offset)
    critics = _select_agents(sorted_critics, num_critics, critic_offset)

    logger.info(
        f"Society inference: {len(actors)} actors, {len(critics)} critics, "
        f"strategy={voting_strategy}, router_top_k={router_top_k}, "
        f"uniform={router_uniform}"
    )

    # Run multi-agent deliberation
    router = CriticRouter(top_k=router_top_k, uniform_weights=router_uniform)
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
        return _majority_vote(answers, task_type)
    elif strategy == "best_actor":
        return _best_actor(answers, result)
    elif strategy == "weighted":
        return _weighted_vote(answers, result, task_type)
    else:
        logger.warning(f"Unknown strategy {strategy}, using majority_vote")
        return _majority_vote(answers, task_type)


def _majority_vote(
    answers: dict[str, str],
    task_type: str = "yes_no",
) -> tuple[str, float]:
    """Majority voting across Actor answers with task-aware normalization.

    For math: groups equivalent numeric answers (e.g., "42" and "42.0").
    For yes_no/multiple_choice: normalizes case and format.
    """
    if not answers:
        return "", 0.0
    from src.algorithms.reward import math_answers_equal

    if task_type == "math":
        # Group math answers by equivalence
        groups: dict[str, list[str]] = {}  # representative -> [originals]
        actor_to_group: dict[str, str] = {}
        for name, ans in answers.items():
            matched_key = None
            for key in groups:
                if math_answers_equal(ans, key):
                    matched_key = key
                    break
            if matched_key:
                groups[matched_key].append(ans)
                actor_to_group[name] = matched_key
            else:
                groups[ans] = [ans]
                actor_to_group[name] = ans

        # Find largest group
        best_key = max(groups, key=lambda k: len(groups[k]))
        confidence = len(groups[best_key]) / len(answers)
        return best_key, confidence
    else:
        # Standard string-based counting (already works for yes_no/multiple_choice)
        counter = Counter(answers.values())
        best, count = counter.most_common(1)[0]
        confidence = count / len(answers)
        return best, confidence


def _best_actor(
    answers: dict[str, str],
    result: MultiDeliberationResult,
) -> tuple[str, float]:
    """Select answer with the strongest agreement across Actors.

    Counts how many Actors agree on each answer and picks the one
    with the most support.  Ties are broken by Critic confidence:
    among tied answers, pick the one whose supporting Actor(s) received
    the highest-average Critic confidence (i.e., the Critics were most
    certain in their feedback about those Actors).
    """
    if not answers:
        return "", 0.0

    # Count agreement per answer value
    answer_votes: dict[str, list[str]] = {}  # answer -> [actor_names]
    for actor_name, answer in answers.items():
        answer_votes.setdefault(answer, []).append(actor_name)

    # Find the answer(s) with the most votes
    max_votes = max(len(v) for v in answer_votes.values())
    top_answers = [a for a, v in answer_votes.items() if len(v) == max_votes]

    if len(top_answers) == 1:
        best_answer = top_answers[0]
        confidence = max_votes / len(answers)
        return best_answer, confidence

    # Tie-break: use average Critic confidence for the tied answers' actors
    actor_confs = _collect_avg_critic_confidences(result)
    best_answer = max(top_answers, key=lambda a: sum(actor_confs.get(n, 0.5) for n in answer_votes[a]))
    confidence = max_votes / len(answers)
    return best_answer, confidence


def _collect_avg_critic_confidences(result: MultiDeliberationResult) -> dict[str, float]:
    """Collect average Critic confidence per Actor from the last round.

    This is the Critics' self-reported confidence in their feedback,
    used only as a tie-breaker for _best_actor — not as a primary voting signal.
    """
    actor_confs: dict[str, float] = {}
    if result.rounds:
        last_round = result.rounds[-1]
        for actor_name, routed in last_round.routed_feedbacks.items():
            critic_confs = [fb.confidence for fb in routed.raw_feedbacks
                           if fb.critic_name in routed.selected_critics]
            actor_confs[actor_name] = (sum(critic_confs) / len(critic_confs)) if critic_confs else 0.5
    return actor_confs


def _collect_answer_correctness(result: MultiDeliberationResult) -> dict[str, float]:
    """Collect Critics' answer-correctness judgment per Actor from the last round.

    Each Critic outputs [Answer_Correct: yes/no].  For each Actor, we count
    the fraction of selected Critics that judged the Actor's answer as correct.
    This is used by _weighted_vote as the primary voting weight.

    Falls back to Critic confidence (feedback-quality proxy) when
    answer-correctness tags are absent (backward compat).
    """
    actor_scores: dict[str, float] = {}
    if result.rounds:
        last_round = result.rounds[-1]
        for actor_name, routed in last_round.routed_feedbacks.items():
            selected_fbs = [fb for fb in routed.raw_feedbacks
                            if fb.critic_name in routed.selected_critics]
            if not selected_fbs:
                actor_scores[actor_name] = 0.5
                continue

            # Prefer answer_correct judgment; fall back to confidence
            has_judgment = any(fb.answer_correct is not None for fb in selected_fbs)
            if has_judgment:
                # Map yes->1.0, no->0.0; skip Critics that didn't judge
                judgments = []
                for fb in selected_fbs:
                    if fb.answer_correct is True:
                        judgments.append(1.0)
                    elif fb.answer_correct is False:
                        judgments.append(0.0)
                    # Skip None (Critic didn't output the tag)
                actor_scores[actor_name] = (sum(judgments) / len(judgments)) if judgments else 0.5
            else:
                # Backward compat: use confidence as proxy
                confs = [fb.confidence for fb in selected_fbs]
                actor_scores[actor_name] = sum(confs) / len(confs)
    return actor_scores


def _weighted_vote(
    answers: dict[str, str],
    result: MultiDeliberationResult,
    task_type: str = "yes_no",
) -> tuple[str, float]:
    """Weighted voting: each Actor's vote is weighted by Critics' correctness judgment.

    This leverages the Critics' [Answer_Correct: yes/no] tags -- Actors whose
    answers were judged correct by more Critics receive higher voting weight.
    Falls back to Critic confidence when answer-correctness tags are absent.

    For math tasks, groups equivalent answers (e.g., "42" and "42.0") using
    math_answers_equal, consistent with _majority_vote.
    """
    if not answers:
        return "", 0.0

    from src.algorithms.reward import math_answers_equal

    # Collect Critics' correctness judgment per Actor
    actor_scores = _collect_answer_correctness(result)

    # Group actors by answer value (math-aware for equivalence)
    answer_votes: dict[str, list[str]] = {}
    actor_to_group: dict[str, str] = {}

    if task_type == "math":
        for actor_name, answer in answers.items():
            matched_key = None
            for key in answer_votes:
                if math_answers_equal(answer, key):
                    matched_key = key
                    break
            if matched_key:
                answer_votes[matched_key].append(actor_name)
                actor_to_group[actor_name] = matched_key
            else:
                answer_votes[answer] = [actor_name]
                actor_to_group[actor_name] = answer
    else:
        for actor_name, answer in answers.items():
            answer_votes.setdefault(answer, []).append(actor_name)
            actor_to_group[actor_name] = answer

    # Weight each answer = sum of its actors' correctness scores
    answer_weights = {
        answer: sum(actor_scores.get(name, 0.5) for name in actor_names)
        for answer, actor_names in answer_votes.items()
    }

    total_weight = sum(answer_weights.values())
    if total_weight == 0:
        # Fallback to majority vote
        best_answer = max(answer_votes, key=lambda a: len(answer_votes[a]))
        return best_answer, len(answer_votes[best_answer]) / len(answers) if answers else 0.0

    best_answer = max(answer_weights, key=lambda a: answer_weights[a])
    confidence = answer_weights[best_answer] / total_weight
    return best_answer, confidence


# ============================================================
# Ablation experiment configurations
# ============================================================

ABLATION_CONFIGS = {
    "A0": {"num_actors": 1, "num_critics": 1, "router_top_k": 1, "router_uniform": False,
            "description": "Base model baseline (no LoRA, no training)"},
    "A1": {"num_actors": 1, "num_critics": 1, "router_top_k": 1, "router_uniform": False,
            "description": "1 trained Actor + 1 trained Critic (ACC-Collab baseline)"},
    "A2": {"num_actors": 3, "num_critics": 1, "router_top_k": 1, "router_uniform": False,
            "description": "3 trained Actors + 1 trained Critic (Actor diversity only)"},
    "A3": {"num_actors": 1, "num_critics": 4, "router_top_k": 2, "router_uniform": False,
            "description": "1 trained Actor + 4 trained Critics + Router (Critic specialization only)"},
    "A4": {"num_actors": 3, "num_critics": 4, "router_top_k": 4, "router_uniform": True,
            "description": "3 trained Actors + 4 trained Critics, uniform weights (no routing)"},
    "A5": {"num_actors": 3, "num_critics": 4, "router_top_k": 2, "router_uniform": False,
            "description": "3 trained Actors + 4 trained Critics + Router (full system)"},
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
                ablation_label=config_name,
            )
            config_results.append(result)

        results[config_name] = config_results
        logger.info(f"  {config_name}: {len(config_results)} samples processed")

    return results
