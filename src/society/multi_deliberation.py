"""
Multi-agent deliberation engine for Diverse Actor-Critic Society.

Implements the single-GPU deliberation with batched inference:
  1. All Actors generate responses (batched into single vLLM call)
  2. All Critics evaluate each Actor response (batched per actor)
  3. Router combines feedback (softmax confidence -> Top-K -> weighted concat)
  4. Feedback is injected into next-round Actor prompts
  5. Majority vote for consensus answer

LoRA support:
  LoRA adapters are loaded dynamically into the vLLM engine via
  the vLLM LoRA API.  Each agent's ``lora_path`` (if set) is used
  to load the adapter before that agent's generation calls.

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
# LoRA management
# ============================================================

# Stable LoRA ID assignment: deterministic mapping from path to integer
_LORA_ID_COUNTER: int = 0
_LORA_ID_CACHE: dict[str, int] = {}


def _get_stable_lora_id(lora_path: str) -> int:
    """Get a stable, unique integer ID for a LoRA path."""
    global _LORA_ID_COUNTER, _LORA_ID_CACHE
    if lora_path not in _LORA_ID_CACHE:
        _LORA_ID_COUNTER += 1
        _LORA_ID_CACHE[lora_path] = _LORA_ID_COUNTER
    return _LORA_ID_CACHE[lora_path]


class LoRAError(RuntimeError):
    """Raised when a LoRA adapter cannot be loaded or used."""
    pass


def _resolve_adapter_path(path: str) -> str:
    """Resolve a LoRA adapter path.

    _dpo_runner.py saves:
      - LoRA adapter  -> output_dir + "_adapter/"
      - merged model  -> output_dir/

    Callers typically hold the merged-model path.  This function
    automatically discovers the adapter directory next to it.

    Raises LoRAError if no valid adapter directory is found.
    """
    p = Path(path)

    # Case 1: path already points to a valid adapter directory
    if (p / "adapter_config.json").exists():
        return str(p)

    # Case 2: try _adapter suffix (convention from _dpo_runner.py)
    adapter_path = Path(str(p) + "_adapter")
    if (adapter_path / "adapter_config.json").exists():
        logger.info(f"Resolved adapter path: {adapter_path} (from {path})")
        return str(adapter_path)

    raise LoRAError(
        f"LoRA adapter not found.  Tried:\n"
        f"  1. {p}/adapter_config.json\n"
        f"  2. {adapter_path}/adapter_config.json\n"
        f"If '{p}' is a merged model directory, the adapter should be at "
        f"'{adapter_path}'.  Make sure _dpo_runner.py has been run and both "
        f"directories exist."
    )


def _load_lora_adapter(engine: Any, lora_path: str) -> Any:
    """Create a LoRARequest for the given adapter path.

    Validates that the adapter directory exists and contains the required
    files.  Raises LoRAError if the adapter is missing or invalid.

    Returns a LoRARequest object (never None for non-empty paths).
    """
    if not lora_path:
        return None

    # Validate engine supports LoRA
    if hasattr(engine, 'supports_lora') and not engine.supports_lora:
        raise LoRAError(
            f"Agent requires LoRA adapter at '{lora_path}', but the "
            f"VLLMInference engine was created without enable_lora=True.  "
            f"Re-create the engine with enable_lora=True, max_loras=<N>, "
            f"max_lora_rank=256."
        )

    # Resolve adapter path (handles _adapter suffix convention)
    resolved_path = _resolve_adapter_path(lora_path)

    try:
        from vllm import LoRARequest
        lora_name = Path(resolved_path).name or "adapter"
        lora_id = _get_stable_lora_id(resolved_path)
        lora_request = LoRARequest(lora_name, lora_id=lora_id, lora_path=resolved_path)
        logger.info(f"Created LoRARequest: name={lora_name}, id={lora_id}, path={resolved_path}")
        return lora_request
    except ImportError as e:
        raise LoRAError(
            f"vllm.LoRARequest import failed: {e}.  "
            f"Ensure vllm >= 0.6 is installed for LoRA support."
        ) from e
    except Exception as e:
        raise LoRAError(
            f"Failed to create LoRARequest for '{resolved_path}': {e}"
        ) from e


# ============================================================
# Multi-agent deliberation (single GPU, batched inference)
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
    Multi-agent deliberation on a single GPU with batched inference.

    For each round:
    1. All Actors generate responses (batched into 1 vLLM call)
    2. All Critics evaluate each Actor (batched: M critics x 1 actor = 1 call)
    3. Router combines Critic feedback
    4. Feedback is injected into next-round Actor prompts
    5. Consensus via majority vote

    LoRA adapters are loaded via LoRARequest objects passed to generate().

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

    if num_rounds <= 0:
        logger.warning("num_rounds <= 0, returning empty result")
        return MultiDeliberationResult(
            rounds=[],
            final_answers={},
            consensus_answer=None,
            consensus_confidence=0.0,
            metadata={"num_actors": len(actors), "num_critics": len(critics)},
        )

    task_type = sample.get("task_type", "yes_no")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    lora_paths = lora_paths or {}

    # Check which agents require LoRA and validate engine support
    agents_needing_lora = []
    for agent in list(actors) + list(critics):
        lora_path = lora_paths.get(agent.name, agent.lora_path)
        if lora_path:
            agents_needing_lora.append((agent.name, lora_path))

    if agents_needing_lora:
        engine_supports_lora = (
            hasattr(inference_engine, 'supports_lora')
            and inference_engine.supports_lora
        )
        if not engine_supports_lora:
            agent_list = ", ".join(f"{n} ({p})" for n, p in agents_needing_lora)
            raise LoRAError(
                f"{len(agents_needing_lora)} agents require LoRA adapters "
                f"but the inference engine does not have LoRA enabled.\n"
                f"  Agents: {agent_list}\n"
                f"  Fix: create VLLMInference with enable_lora=True, "
                f"max_loras={len(agents_needing_lora)}, max_lora_rank=256"
            )

    # Pre-create LoRA request objects for each agent (once, reuse across rounds)
    lora_requests: dict[str, Any] = {}
    for agent in list(actors) + list(critics):
        lora_path = lora_paths.get(agent.name, agent.lora_path)
        if lora_path:
            lora_req = _load_lora_adapter(inference_engine, lora_path)
            if lora_req is not None:
                lora_requests[agent.name] = lora_req
                logger.info(f"Loaded LoRA for {agent.name}: {lora_path}")

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

        # ---- Step 1: Batch all Actor responses into 1 vLLM call ----
        # Check crash recovery first
        uncached_actors = []
        for actor in actors:
            ckpt_path = ckpt_dir / f"round_{round_num}" / f"actor_{actor.name}.json" if ckpt_dir else None
            cached = _load_json(ckpt_path) if ckpt_path else None
            if cached and "actor_response" in cached:
                actor_response = cached["actor_response"]
                actor_answer = extract_answer(actor_response, task_type)
                round_data.actor_responses[actor.name] = actor_response
                round_data.actor_answers[actor.name] = actor_answer
                actor_histories[actor.name].append(actor_response)
                logger.info(f"[Recovery] Loaded cached response for {actor.name} round {round_num}")
            else:
                uncached_actors.append(actor)

        if uncached_actors:
            # Build all actor prompts
            actor_prompts = []
            for actor in uncached_actors:
                prompt = _build_actor_prompt(
                    actor, sample, dataset_name, round_num,
                    actor_histories[actor.name],
                )
                actor_prompts.append(prompt)

            # Generate with LoRA if available
            actor_responses_batch = _generate_with_lora(
                inference_engine,
                actor_prompts,
                uncached_actors,
                lora_requests,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Store results
            for actor, response in zip(uncached_actors, actor_responses_batch):
                response = response if isinstance(response, str) else str(response)
                actor_answer = extract_answer(response, task_type)
                round_data.actor_responses[actor.name] = response
                round_data.actor_answers[actor.name] = actor_answer
                actor_histories[actor.name].append(response)

                if ckpt_dir:
                    ckpt_path = ckpt_dir / f"round_{round_num}" / f"actor_{actor.name}.json"
                    _atomic_write_json(ckpt_path, {"actor_response": response})

        # ---- Step 2: Batch all Critic evaluations ----
        for actor in actors:
            actor_response = round_data.actor_responses[actor.name]
            round_data.critic_feedbacks[actor.name] = {}

            # Check cache for all critics
            uncached_critics = []
            for critic in critics:
                ckpt_path = (ckpt_dir / f"round_{round_num}" /
                             f"critic_{critic.name}_for_{actor.name}.json" if ckpt_dir else None)
                cached = _load_json(ckpt_path) if ckpt_path else None
                if cached and "critic_response" in cached:
                    round_data.critic_feedbacks[actor.name][critic.name] = cached["critic_response"]
                else:
                    uncached_critics.append(critic)

            if uncached_critics:
                # Build all critic prompts for this actor
                critic_prompts = [
                    _build_critic_prompt(c, sample, dataset_name, actor_response)
                    for c in uncached_critics
                ]

                # Generate with LoRA if available
                critic_responses_batch = _generate_with_lora(
                    inference_engine,
                    critic_prompts,
                    uncached_critics,
                    lora_requests,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                for critic, response in zip(uncached_critics, critic_responses_batch):
                    response = response if isinstance(response, str) else str(response)
                    round_data.critic_feedbacks[actor.name][critic.name] = response

                    if ckpt_dir:
                        ckpt_path = (ckpt_dir / f"round_{round_num}" /
                                     f"critic_{critic.name}_for_{actor.name}.json")
                        _atomic_write_json(ckpt_path, {"critic_response": response})

            # ---- Step 3: Router combines feedback ----
            critic_feedbacks = []
            for critic in critics:
                resp = round_data.critic_feedbacks[actor.name].get(critic.name, "")
                fb = build_critic_feedback(critic, resp)
                critic_feedbacks.append(fb)
            routed = router.route(critic_feedbacks)
            round_data.routed_feedbacks[actor.name] = routed

        # ---- Step 4: Inject routed feedback into actor histories for next round ----
        # This is critical: Actors must see Critic feedback to improve their answers
        for actor in actors:
            routed = round_data.routed_feedbacks.get(actor.name)
            if routed and routed.feedback_text:
                actor_histories[actor.name].append(
                    f"[Critic Feedback]\n{routed.feedback_text}"
                )

        # ---- Step 5: Majority vote for consensus ----
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
# LoRA-aware generation helper
# ============================================================

def _generate_with_lora(
    engine: Any,
    prompts: list[str],
    agents: list[AgentConfig],
    lora_requests: dict[str, Any],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[str]:
    """Generate responses with per-agent LoRA adapters.

    Groups agents by their LoRA request (or None for base model) and batches
    each group into a single vLLM call.

    Raises LoRAError if a LoRA adapter is required but cannot be used.
    """
    # Group agents by their LoRA request (or None for base model)
    groups: dict[Optional[Any], list[tuple[int, str]]] = {}  # lora_req -> [(idx, prompt)]
    for i, (prompt, agent) in enumerate(zip(prompts, agents)):
        lora_req = lora_requests.get(agent.name)
        groups.setdefault(lora_req, []).append((i, prompt))

    results: list[Optional[str]] = [None] * len(prompts)

    for lora_req, items in groups.items():
        batch_prompts = [p for _, p in items]
        batch_indices = [idx for idx, _ in items]

        if lora_req is not None:
            # Use vLLM's LoRA-aware generation
            try:
                if hasattr(engine, 'generate_with_lora'):
                    batch_results = engine.generate_with_lora(
                        batch_prompts, lora_req,
                        max_tokens=max_tokens, temperature=temperature,
                    )
                elif hasattr(engine, '_llm') and engine._llm is not None:
                    from vllm import SamplingParams
                    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, n=1)
                    outputs = engine._llm.generate(batch_prompts, params, lora_request=lora_req)
                    batch_results = [c.text for o in outputs for c in o.outputs]
                else:
                    raise LoRAError(
                        f"Engine does not support LoRA generation for adapter "
                        f"'{lora_req.lora_path}'.  The VLLMInference engine must "
                        f"be created with enable_lora=True."
                    )
            except LoRAError:
                raise
            except Exception as e:
                agent_names = [agents[idx].name for idx in batch_indices]
                raise LoRAError(
                    f"LoRA generation failed for agents {agent_names} with "
                    f"adapter '{getattr(lora_req, 'lora_path', lora_req)}': {e}"
                ) from e
        else:
            # No LoRA: use standard generate
            batch_results = engine.generate(
                batch_prompts, max_tokens=max_tokens, temperature=temperature,
            )

        for idx, result in zip(batch_indices, batch_results):
            results[idx] = result if isinstance(result, str) else str(result)

    return results


# ============================================================
# Prompt builders (separated for testability and reuse)
# ============================================================

def _build_actor_prompt(
    actor: AgentConfig,
    sample: dict,
    dataset_name: str,
    round_num: int,
    previous_responses: list[str],
) -> str:
    """Build prompt for an Actor response.

    For round > 0, Critic feedback is already appended to previous_responses
    by the deliberation loop, so it will be included via the template.
    """
    from src.prompts.templates import get_prompt_template, PromptType

    if round_num == 0:
        prompt_type = PromptType.SINGLE_SHOT
    else:
        prompt_type = PromptType.DELIBERATION_ACTOR

    template = get_prompt_template(dataset_name, prompt_type)

    # Build responses text from previous rounds (includes Critic feedback)
    if previous_responses:
        responses_text = "\n\n".join(previous_responses)
    else:
        responses_text = ""

    prompt = template.format(
        question=sample.get("question", ""),
        passage=sample.get("passage", ""),
        responses_text=responses_text,
        choice_a=sample.get("choices", ["", "", "", ""])[0] if len(sample.get("choices", [])) >= 1 else "",
        choice_b=sample.get("choices", ["", "", "", ""])[1] if len(sample.get("choices", [])) >= 2 else "",
        choice_c=sample.get("choices", ["", "", "", ""])[2] if len(sample.get("choices", [])) >= 3 else "",
        choice_d=sample.get("choices", ["", "", "", ""])[3] if len(sample.get("choices", [])) >= 4 else "",
    )

    # Prepend actor's style-specific system prompt
    if actor.system_prompt:
        prompt = f"{actor.system_prompt}\n\n{prompt}"

    return prompt


def _build_critic_prompt(
    critic: AgentConfig,
    sample: dict,
    dataset_name: str,
    actor_response: str,
) -> str:
    """Build prompt for a Critic feedback."""
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

    return prompt


# Backward-compatible wrappers (used by society_trainer.py)
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
    """Generate a single Actor response (backward-compatible wrapper)."""
    prompt = _build_actor_prompt(actor, sample, dataset_name, round_num, previous_responses)
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
    """Generate Critic feedback for an Actor response (backward-compatible wrapper)."""
    prompt = _build_critic_prompt(critic, sample, dataset_name, actor_response)
    result = engine.generate_single(prompt, max_tokens=max_tokens, temperature=temperature)
    return result if isinstance(result, str) else str(result)
