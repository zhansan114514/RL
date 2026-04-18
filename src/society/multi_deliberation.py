"""
Multi-agent deliberation engine for Diverse Actor-Critic Society.

Implements the single-GPU sequential deliberation from the experiment plan:
  1. All Actors generate responses (sequential LoRA load/unload)
  2. All Critics evaluate each Actor response
  3. Router combines feedback (softmax confidence → Top-K → weighted concat)
  4. Actors receive routed feedback → next round revision
  5. Majority vote for consensus answer

Crash recovery: each model call persisted to disk, skip on restart.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.algorithms.reward import extract_answer
from src.society.agent_registry import AgentConfig, AgentRole
from src.society.router import CriticRouter, build_critic_feedback, RoutedFeedback

logger = logging.getLogger(__name__)


# ============================================================
# Data classes
# ============================================================

@dataclass
class DeliberationRound:
    """One round of multi-agent deliberation."""
    round_num: int
    actor_responses: dict[str, str]  # actor_name -> response
    actor_answers: dict[str, Optional[str]]  # actor_name -> extracted answer
    critic_feedbacks: dict[str, dict[str, str]]  # actor_name -> {critic_name: feedback}
    routed_feedbacks: dict[str, RoutedFeedback]  # actor_name -> routed
    consensus_answer: Optional[str] = None


@dataclass
class MultiDeliberationResult:
    """Complete result of multi-agent deliberation."""
    rounds: list[DeliberationRound]
    final_answers: dict[str, Optional[str]]  # actor_name -> final answer
    consensus_answer: Optional[str]
    consensus_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # Compatibility aliases for scripts that use different field names
    @property
    def final_answer(self) -> Optional[str]:
        return self.consensus_answer

    @property
    def confidence(self) -> float:
        return self.consensus_confidence

    @property
    def individual_results(self) -> list[dict]:
        """Return per-actor results for backward compatibility."""
        results = []
        for actor_name, answer in self.final_answers.items():
            trajectory = []
            for r in self.rounds:
                if actor_name in r.actor_responses:
                    trajectory.append({
                        "actor_response": r.actor_responses[actor_name],
                        "actor_answer": r.actor_answers.get(actor_name),
                    })
                    if actor_name in r.critic_feedbacks:
                        for cn, fb in r.critic_feedbacks[actor_name].items():
                            trajectory[-1][f"critic_{cn}_feedback"] = fb
            results.append({
                "actor_name": actor_name,
                "final_answer": answer,
                "trajectory": trajectory,
            })
        return results


# ============================================================
# Atomic disk persistence for crash recovery
# ============================================================

def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON if exists."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


# ============================================================
# LoRA management helpers
# ============================================================

def _load_lora_adapter(engine: Any, lora_path: str) -> Optional[str]:
    """Load a LoRA adapter into the vLLM engine.

    Returns the LoRA request ID if successful, None otherwise.
    """
    if not lora_path:
        return None
    try:
        # vLLM supports dynamic LoRA loading via load_lora_adapter
        if hasattr(engine, 'load_lora_adapter'):
            return engine.load_lora_adapter(lora_path)
        elif hasattr(engine, '_engine') and hasattr(engine._engine, 'add_lora'):
            return engine._engine.add_lora(lora_path)
    except Exception as e:
        logger.warning(f"Failed to load LoRA adapter from {lora_path}: {e}")
    return None


def _unload_lora_adapter(engine: Any, lora_request_id: Optional[str]) -> None:
    """Unload a LoRA adapter from the vLLM engine."""
    if not lora_request_id:
        return
    try:
        if hasattr(engine, 'unload_lora_adapter'):
            engine.unload_lora_adapter(lora_request_id)
        elif hasattr(engine, '_engine') and hasattr(engine._engine, 'remove_lora'):
            engine._engine.remove_lora(lora_request_id)
    except Exception as e:
        logger.warning(f"Failed to unload LoRA adapter {lora_request_id}: {e}")


# ============================================================
# Multi-agent deliberation (single GPU)
# ============================================================

def multi_agent_deliberate_single_gpu(
    inference_engine: Any,  # VLLMInference
    actors: list[AgentConfig],
    critics: list[AgentConfig],
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
    router: Optional[CriticRouter] = None,
    checkpoint_dir: Optional[str] = None,
    lora_paths: Optional[dict[str, str]] = None,
) -> MultiDeliberationResult:
    """
    Multi-agent deliberation on a single GPU (sequential execution).

    For each round:
    1. All Actors generate responses (sequentially with LoRA switching)
    2. All Critics evaluate each Actor (sequentially with LoRA switching)
    3. Router combines Critic feedback
    4. Consensus via majority vote

    Args:
        inference_engine: VLLMInference instance (base model).
        actors: List of Actor AgentConfigs.
        critics: List of Critic AgentConfigs.
        sample: Standardized sample dict.
        dataset_name: Dataset name for prompt templates.
        num_rounds: Number of deliberation rounds.
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        router: CriticRouter for combining feedback.
        checkpoint_dir: Directory for crash recovery persistence.
        lora_paths: Optional dict mapping agent_name -> LoRA adapter path.

    Returns:
        MultiDeliberationResult with complete deliberation history.
    """
    if router is None:
        router = CriticRouter(top_k=2)

    task_type = sample.get("task_type", "yes_no")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    lora_paths = lora_paths or {}

    rounds: list[DeliberationRound] = []
    # Track previous responses per actor (for deliberation prompts)
    actor_histories: dict[str, list[str]] = {a.name: [] for a in actors}

    for round_num in range(num_rounds):
        round_data = DeliberationRound(
            round_num=round_num,
            actor_responses={},
            actor_answers={},
            critic_feedbacks={},
            routed_feedbacks={},
        )

        # ---- Step 1: All Actors generate responses ----
        for actor in actors:
            ckpt_path = ckpt_dir / f"round_{round_num}" / f"actor_{actor.name}.json" if ckpt_dir else None

            # Check crash recovery
            cached = _load_json(ckpt_path) if ckpt_path else None
            if cached and "actor_response" in cached:
                actor_response = cached["actor_response"]
                logger.info(f"[Recovery] Loaded cached response for {actor.name} round {round_num}")
            else:
                # Load LoRA adapter if available
                actor_lora_id = _load_lora_adapter(
                    inference_engine, lora_paths.get(actor.name, actor.lora_path or "")
                )

                try:
                    actor_response = _generate_actor_response(
                        engine=inference_engine,
                        actor=actor,
                        sample=sample,
                        dataset_name=dataset_name,
                        round_num=round_num,
                        previous_responses=actor_histories[actor.name],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                finally:
                    # Unload LoRA after generation
                    _unload_lora_adapter(inference_engine, actor_lora_id)

                # Persist
                if ckpt_path:
                    _atomic_write_json(ckpt_path, {"actor_response": actor_response})

            actor_answer = extract_answer(actor_response, task_type)
            round_data.actor_responses[actor.name] = actor_response
            round_data.actor_answers[actor.name] = actor_answer
            actor_histories[actor.name].append(actor_response)

        # ---- Step 2: All Critics evaluate each Actor ----
        for actor in actors:
            actor_response = round_data.actor_responses[actor.name]
            round_data.critic_feedbacks[actor.name] = {}

            critic_feedbacks = []
            for critic in critics:
                ckpt_path = (ckpt_dir / f"round_{round_num}" /
                             f"critic_{critic.name}_for_{actor.name}.json" if ckpt_dir else None)

                cached = _load_json(ckpt_path) if ckpt_path else None
                if cached and "critic_response" in cached:
                    critic_response = cached["critic_response"]
                else:
                    # Load Critic's LoRA adapter
                    critic_lora_id = _load_lora_adapter(
                        inference_engine, lora_paths.get(critic.name, critic.lora_path or "")
                    )

                    try:
                        critic_response = _generate_critic_feedback(
                            engine=inference_engine,
                            critic=critic,
                            sample=sample,
                            dataset_name=dataset_name,
                            actor_response=actor_response,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                    finally:
                        _unload_lora_adapter(inference_engine, critic_lora_id)

                    if ckpt_path:
                        _atomic_write_json(ckpt_path, {"critic_response": critic_response})

                round_data.critic_feedbacks[actor.name][critic.name] = critic_response

                # Build CriticFeedback for routing
                fb = build_critic_feedback(critic, critic_response)
                critic_feedbacks.append(fb)

            # ---- Step 3: Router combines feedback ----
            routed = router.route(critic_feedbacks)
            round_data.routed_feedbacks[actor.name] = routed

        # ---- Step 4: Majority vote for consensus ----
        answers = [a for a in round_data.actor_answers.values() if a is not None]
        if answers:
            counter = Counter(answers)
            most_common = counter.most_common(1)[0]
            round_data.consensus_answer = most_common[0]

        rounds.append(round_data)

    # Final answers from last round
    final_round = rounds[-1]
    final_answers = dict(final_round.actor_answers)
    consensus_answer = final_round.consensus_answer

    # Consensus confidence
    if consensus_answer and final_answers:
        agree_count = sum(1 for a in final_answers.values() if a == consensus_answer)
        consensus_confidence = agree_count / len(final_answers)
    else:
        consensus_confidence = 0.0

    return MultiDeliberationResult(
        rounds=rounds,
        final_answers=final_answers,
        consensus_answer=consensus_answer,
        consensus_confidence=consensus_confidence,
        metadata={"num_actors": len(actors), "num_critics": len(critics)},
    )


# Keep old name for backward compat
multi_agent_deliberate = multi_agent_deliberate_single_gpu


# ============================================================
# Helper: Actor response generation
# ============================================================

def _generate_actor_response(
    engine: Any,
    actor: AgentConfig,
    sample: dict,
    dataset_name: str,
    round_num: int,
    previous_responses: list[str],
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate a single Actor response."""
    from src.prompts.templates import get_prompt_template, PromptType

    if round_num == 0:
        prompt_type = PromptType.SINGLE_SHOT
    else:
        prompt_type = PromptType.DELIBERATION_ACTOR

    template = get_prompt_template(dataset_name, prompt_type)

    # Build responses text from previous rounds
    responses_text = ""
    if previous_responses:
        responses_text = "\n".join(
            f"\nPerson {i+1}: {r}" for i, r in enumerate(previous_responses[-3:])
        )

    prompt = template.format(
        question=sample.get("question", ""),
        passage=sample.get("passage", ""),
        responses_text=responses_text,
        # MC placeholders (empty for non-MC)
        choice_a=sample.get("choices", ["", "", "", ""])[0] if len(sample.get("choices", [])) >= 1 else "",
        choice_b=sample.get("choices", ["", "", "", ""])[1] if len(sample.get("choices", [])) >= 2 else "",
        choice_c=sample.get("choices", ["", "", "", ""])[2] if len(sample.get("choices", [])) >= 3 else "",
        choice_d=sample.get("choices", ["", "", "", ""])[3] if len(sample.get("choices", [])) >= 4 else "",
    )

    # Prepend actor's style-specific system prompt
    if actor.system_prompt:
        prompt = f"{actor.system_prompt}\n\n{prompt}"

    result = engine.generate_single(prompt, max_tokens=max_tokens, temperature=temperature)
    return result if isinstance(result, str) else str(result)


def _generate_critic_feedback(
    engine: Any,
    critic: AgentConfig,
    sample: dict,
    dataset_name: str,
    actor_response: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate Critic feedback for an Actor response."""
    from src.prompts.templates import get_prompt_template, PromptType

    template = get_prompt_template(dataset_name, PromptType.DELIBERATION_CRITIC)

    prompt = template.format(
        question=sample.get("question", ""),
        passage=sample.get("passage", ""),
        actor_response=actor_response,
        choice_a=sample.get("choices", ["", "", "", ""])[0] if len(sample.get("choices", [])) >= 1 else "",
        choice_b=sample.get("choices", ["", "", "", ""])[1] if len(sample.get("choices", [])) >= 2 else "",
        choice_c=sample.get("choices", ["", "", "", ""])[2] if len(sample.get("choices", [])) >= 3 else "",
        choice_d=sample.get("choices", ["", "", "", ""])[3] if len(sample.get("choices", [])) >= 4 else "",
    )

    if critic.system_prompt:
        prompt = f"{critic.system_prompt}\n\n{prompt}"

    result = engine.generate_single(prompt, max_tokens=max_tokens, temperature=temperature)
    return result if isinstance(result, str) else str(result)
