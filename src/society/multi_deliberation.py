"""Natural multi-agent deliberation engine."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.evaluation.answer_resolution import normalize_task_answer
from src.parsing.answer_extractor import extract_answer
from src.prompts.prompt_builder import build_actor_prompt, build_critic_feedback_prompt
from src.society.agent_registry import AgentConfig
from src.society.router import CriticFeedback, CriticRouter, build_critic_feedback

logger = logging.getLogger(__name__)


@dataclass
class DeliberationRound:
    """One natural deliberation round."""

    round_num: int
    actor_responses: dict[str, str]
    actor_answers: dict[str, Optional[str]]
    actor_answer_sources: dict[str, str]
    actor_parse_confidence: dict[str, float]
    critic_raw_responses: dict[str, dict[str, str]]
    critic_feedbacks: dict[str, dict[str, CriticFeedback]]
    routed_feedbacks: dict[str, list[CriticFeedback]]
    routed_feedback_texts: dict[str, str] = field(default_factory=dict)
    consensus_answer: Optional[str] = None
    consensus_confidence: float = 0.0
    consensus_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class MultiDeliberationResult:
    """Complete result of multi-agent deliberation."""

    rounds: list[DeliberationRound]
    final_answers: dict[str, Optional[str]]
    consensus_answer: Optional[str]
    consensus_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_answer(self) -> Optional[str]:
        return self.consensus_answer

    @property
    def confidence(self) -> float:
        return self.consensus_confidence

    @property
    def individual_results(self) -> list[dict]:
        results = []
        for actor_name, answer in self.final_answers.items():
            trajectory = []
            for round_data in self.rounds:
                if actor_name not in round_data.actor_responses:
                    continue
                item = {
                    "actor_response": round_data.actor_responses[actor_name],
                    "actor_answer": round_data.actor_answers.get(actor_name),
                    "answer_source": round_data.actor_answer_sources.get(actor_name),
                    "parse_confidence": round_data.actor_parse_confidence.get(actor_name),
                }
                for critic_name, fb in round_data.critic_feedbacks.get(actor_name, {}).items():
                    item[f"critic_{critic_name}_feedback"] = fb.critique
                trajectory.append(item)
            results.append({
                "actor_name": actor_name,
                "final_answer": answer,
                "trajectory": trajectory,
            })
        return results


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON if it exists and is valid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


_LORA_ID_COUNTER = 0
_LORA_ID_CACHE: dict[str, int] = {}


def _get_stable_lora_id(lora_path: str) -> int:
    global _LORA_ID_COUNTER
    if lora_path not in _LORA_ID_CACHE:
        _LORA_ID_COUNTER += 1
        _LORA_ID_CACHE[lora_path] = _LORA_ID_COUNTER
    return _LORA_ID_CACHE[lora_path]


class LoRAError(RuntimeError):
    """Raised when a LoRA adapter cannot be loaded or used."""


def _normalize_vote_answer(answer: Optional[str], task_type: str) -> Optional[str]:
    return normalize_task_answer(answer, task_type)


def _extract_actor_answer(response: str, task_type: str) -> tuple[Optional[str], str, float]:
    extracted = extract_answer(response, task_type)
    answer = _normalize_vote_answer(extracted.answer, task_type)
    return answer, extracted.source, extracted.confidence


def _critic_aware_consensus(
    round_data: DeliberationRound,
    task_type: str,
) -> tuple[Optional[str], float, dict[str, float]]:
    """Consensus from Actor votes plus Critic judgement signals."""
    weights: dict[str, float] = {}

    for answer in round_data.actor_answers.values():
        normalized = _normalize_vote_answer(answer, task_type)
        if normalized:
            weights[normalized] = weights.get(normalized, 0.0) + 1.0

    for actor_name, feedbacks in round_data.critic_feedbacks.items():
        actor_answer = _normalize_vote_answer(round_data.actor_answers.get(actor_name), task_type)
        for feedback in feedbacks.values():
            if feedback.confidence is None:
                continue
            if feedback.answer_correct == "yes" and actor_answer:
                weights[actor_answer] = weights.get(actor_answer, 0.0) + feedback.confidence
            elif feedback.answer_correct == "no":
                suggested = _normalize_vote_answer(feedback.suggested_answer, task_type)
                if suggested:
                    weights[suggested] = weights.get(suggested, 0.0) + feedback.confidence

    if not weights:
        return None, 0.0, {}

    actor_counts = Counter(
        answer
        for answer in (
            _normalize_vote_answer(a, task_type)
            for a in round_data.actor_answers.values()
        )
        if answer
    )
    best_answer, best_weight = max(
        weights.items(),
        key=lambda item: (item[1], actor_counts.get(item[0], 0), item[0]),
    )
    total = sum(weights.values())
    confidence = best_weight / total if total > 0 else 0.0
    return best_answer, confidence, weights


def _resolve_adapter_path(path: str) -> str:
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    adapter_path = Path(str(p) + "_adapter")
    if (adapter_path / "adapter_config.json").exists():
        logger.info("Resolved adapter path: %s (from %s)", adapter_path, path)
        return str(adapter_path)
    raise LoRAError(
        f"LoRA adapter not found. Tried {p}/adapter_config.json and "
        f"{adapter_path}/adapter_config.json."
    )


def _load_lora_adapter(engine: Any, lora_path: str) -> Any:
    if not lora_path:
        return None
    if hasattr(engine, "supports_lora") and not engine.supports_lora:
        raise LoRAError(
            f"Agent requires LoRA adapter at '{lora_path}', but the inference "
            "engine was created without LoRA support."
        )
    resolved_path = _resolve_adapter_path(lora_path)
    try:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            from vllm import LoRARequest
        lora_name = Path(resolved_path).name or "adapter"
        lora_id = _get_stable_lora_id(resolved_path)
        return LoRARequest(lora_name, lora_id, resolved_path)
    except ImportError as e:
        raise LoRAError(f"vllm.LoRARequest import failed: {e}") from e
    except Exception as e:
        raise LoRAError(f"Failed to create LoRARequest for '{resolved_path}': {e}") from e


def _generate_with_lora(
    engine: Any,
    prompts: list[str],
    agents: list[AgentConfig],
    lora_requests: dict[str, Any],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[str]:
    """Generate responses with per-agent LoRA adapters."""
    groups: dict[Optional[Any], list[tuple[int, str]]] = {}
    for idx, (prompt, agent) in enumerate(zip(prompts, agents)):
        groups.setdefault(lora_requests.get(agent.name), []).append((idx, prompt))

    results: list[Optional[str]] = [None] * len(prompts)
    for lora_req, items in groups.items():
        batch_prompts = [prompt for _, prompt in items]
        batch_indices = [idx for idx, _ in items]
        if lora_req is not None:
            try:
                if hasattr(engine, "generate_with_lora"):
                    batch_results = engine.generate_with_lora(
                        batch_prompts,
                        lora_req,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                elif hasattr(engine, "_llm") and engine._llm is not None:
                    from vllm import SamplingParams
                    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, n=1)
                    outputs = engine._llm.generate(batch_prompts, params, lora_request=lora_req)
                    batch_results = [choice.text for out in outputs for choice in out.outputs]
                else:
                    raise LoRAError("Engine does not support LoRA generation.")
            except LoRAError:
                raise
            except Exception as e:
                agent_names = [agents[idx].name for idx in batch_indices]
                raise LoRAError(f"LoRA generation failed for agents {agent_names}: {e}") from e
        else:
            batch_results = engine.generate(
                batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        for idx, result in zip(batch_indices, batch_results):
            results[idx] = result if isinstance(result, str) else str(result)

    return [result or "" for result in results]


def _prepare_loras(
    inference_engine: Any,
    actors: list[AgentConfig],
    critics: list[AgentConfig],
    lora_paths: Optional[dict[str, str]],
) -> dict[str, Any]:
    lora_paths = lora_paths or {}
    agents = list(actors) + list(critics)
    needing = []
    for agent in agents:
        path = lora_paths.get(agent.name, agent.lora_path)
        if path:
            needing.append((agent.name, path))
    if needing:
        supports = hasattr(inference_engine, "supports_lora") and inference_engine.supports_lora
        if not supports:
            agent_list = ", ".join(f"{name} ({path})" for name, path in needing)
            raise LoRAError(f"Agents require LoRA but engine does not support it: {agent_list}")

    requests: dict[str, Any] = {}
    for agent in agents:
        path = lora_paths.get(agent.name, agent.lora_path)
        if path:
            requests[agent.name] = _load_lora_adapter(inference_engine, path)
    return requests


def _empty_result(actors: list[AgentConfig], critics: list[AgentConfig]) -> MultiDeliberationResult:
    return MultiDeliberationResult(
        rounds=[],
        final_answers={},
        consensus_answer=None,
        consensus_confidence=0.0,
        metadata={"num_actors": len(actors), "num_critics": len(critics)},
    )


def _actor_feedback_for_next_round(round_data: DeliberationRound, actor_name: str) -> str:
    return round_data.routed_feedback_texts.get(actor_name, "")


def multi_agent_deliberate_single_gpu(
    inference_engine: Any,
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
    """Run natural multi-Actor, multi-Critic deliberation for one sample."""
    if router is None:
        router = CriticRouter(top_k=2)
    if num_rounds <= 0:
        return _empty_result(actors, critics)

    task_type = sample.get("task_type", "yes_no")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    lora_requests = _prepare_loras(inference_engine, actors, critics, lora_paths)
    rounds: list[DeliberationRound] = []
    previous_actor_responses = {actor.name: "" for actor in actors}
    previous_feedback = {actor.name: "" for actor in actors}

    for round_num in range(num_rounds):
        round_data = DeliberationRound(
            round_num=round_num,
            actor_responses={},
            actor_answers={},
            actor_answer_sources={},
            actor_parse_confidence={},
            critic_raw_responses={},
            critic_feedbacks={},
            routed_feedbacks={},
        )

        uncached_actors: list[AgentConfig] = []
        for actor in actors:
            ckpt_path = ckpt_dir / f"round_{round_num}" / f"actor_{actor.name}.json" if ckpt_dir else None
            cached = _load_json(ckpt_path) if ckpt_path else None
            if cached and "actor_response" in cached:
                response = str(cached["actor_response"])
                answer, source, parse_conf = _extract_actor_answer(response, task_type)
                round_data.actor_responses[actor.name] = response
                round_data.actor_answers[actor.name] = answer
                round_data.actor_answer_sources[actor.name] = source
                round_data.actor_parse_confidence[actor.name] = parse_conf
                previous_actor_responses[actor.name] = response
                logger.info("[Recovery] Loaded cached response for %s round %s", actor.name, round_num)
            else:
                uncached_actors.append(actor)

        if uncached_actors:
            prompts = [
                build_actor_prompt(
                    actor,
                    sample,
                    dataset_name,
                    round_num=round_num,
                    previous_actor_response=previous_actor_responses[actor.name],
                    critic_feedback=previous_feedback[actor.name],
                )
                for actor in uncached_actors
            ]
            responses = _generate_with_lora(
                inference_engine,
                prompts,
                uncached_actors,
                lora_requests,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for actor, response in zip(uncached_actors, responses):
                answer, source, parse_conf = _extract_actor_answer(response, task_type)
                round_data.actor_responses[actor.name] = response
                round_data.actor_answers[actor.name] = answer
                round_data.actor_answer_sources[actor.name] = source
                round_data.actor_parse_confidence[actor.name] = parse_conf
                previous_actor_responses[actor.name] = response
                if ckpt_dir:
                    ckpt_path = ckpt_dir / f"round_{round_num}" / f"actor_{actor.name}.json"
                    _atomic_write_json(ckpt_path, {"actor_response": response})

        for actor in actors:
            actor_response = round_data.actor_responses[actor.name]
            round_data.critic_raw_responses[actor.name] = {}
            round_data.critic_feedbacks[actor.name] = {}

            uncached_critics: list[AgentConfig] = []
            for critic in critics:
                ckpt_path = (
                    ckpt_dir / f"round_{round_num}" / f"critic_{critic.name}_for_{actor.name}.json"
                    if ckpt_dir else None
                )
                cached = _load_json(ckpt_path) if ckpt_path else None
                if cached and "critic_response" in cached:
                    response = str(cached["critic_response"])
                    round_data.critic_raw_responses[actor.name][critic.name] = response
                else:
                    uncached_critics.append(critic)

            if uncached_critics:
                prompts = [
                    build_critic_feedback_prompt(critic, sample, dataset_name, actor_response)
                    for critic in uncached_critics
                ]
                responses = _generate_with_lora(
                    inference_engine,
                    prompts,
                    uncached_critics,
                    lora_requests,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                for critic, response in zip(uncached_critics, responses):
                    round_data.critic_raw_responses[actor.name][critic.name] = response
                    if ckpt_dir:
                        ckpt_path = (
                            ckpt_dir / f"round_{round_num}" / f"critic_{critic.name}_for_{actor.name}.json"
                        )
                        _atomic_write_json(ckpt_path, {"critic_response": response})

            feedbacks: list[CriticFeedback] = []
            for critic in critics:
                raw = round_data.critic_raw_responses[actor.name].get(critic.name, "")
                feedback = build_critic_feedback(critic, raw, task_type=task_type)
                round_data.critic_feedbacks[actor.name][critic.name] = feedback
                feedbacks.append(feedback)

            routed = router.route(feedbacks)
            round_data.routed_feedbacks[actor.name] = routed.selected_feedbacks
            round_data.routed_feedback_texts[actor.name] = routed.feedback_text
            previous_feedback[actor.name] = routed.feedback_text

        consensus_answer, consensus_confidence, consensus_weights = _critic_aware_consensus(
            round_data,
            task_type,
        )
        round_data.consensus_answer = consensus_answer
        round_data.consensus_confidence = consensus_confidence
        round_data.consensus_weights = consensus_weights
        rounds.append(round_data)

    final_round = rounds[-1]
    return MultiDeliberationResult(
        rounds=rounds,
        final_answers=dict(final_round.actor_answers),
        consensus_answer=final_round.consensus_answer,
        consensus_confidence=final_round.consensus_confidence,
        metadata={"num_actors": len(actors), "num_critics": len(critics)},
    )


def multi_agent_deliberate_batched_single_gpu(
    inference_engine: Any,
    actors: list[AgentConfig],
    critics: list[AgentConfig],
    samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
    router: Optional[CriticRouter] = None,
    lora_paths: Optional[dict[str, str]] = None,
) -> list[MultiDeliberationResult]:
    """Batched natural deliberation across samples."""
    if not samples:
        return []
    if router is None:
        router = CriticRouter(top_k=2)
    if num_rounds <= 0:
        return [_empty_result(actors, critics) for _ in samples]

    n = len(samples)
    task_types = [sample.get("task_type", "yes_no") for sample in samples]
    lora_requests = _prepare_loras(inference_engine, actors, critics, lora_paths)
    all_rounds: list[list[DeliberationRound]] = [[] for _ in range(n)]
    previous_actor_responses = {actor.name: [""] * n for actor in actors}
    previous_feedback = {actor.name: [""] * n for actor in actors}

    for round_num in range(num_rounds):
        round_start = time.time()
        round_datas = [
            DeliberationRound(
                round_num=round_num,
                actor_responses={},
                actor_answers={},
                actor_answer_sources={},
                actor_parse_confidence={},
                critic_raw_responses={},
                critic_feedbacks={},
                routed_feedbacks={},
            )
            for _ in range(n)
        ]

        for actor in actors:
            prompts = [
                build_actor_prompt(
                    actor,
                    samples[i],
                    dataset_name,
                    round_num=round_num,
                    previous_actor_response=previous_actor_responses[actor.name][i],
                    critic_feedback=previous_feedback[actor.name][i],
                )
                for i in range(n)
            ]
            responses = _generate_with_lora(
                inference_engine,
                prompts,
                [actor] * n,
                lora_requests,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for i, response in enumerate(responses):
                answer, source, parse_conf = _extract_actor_answer(response, task_types[i])
                round_datas[i].actor_responses[actor.name] = response
                round_datas[i].actor_answers[actor.name] = answer
                round_datas[i].actor_answer_sources[actor.name] = source
                round_datas[i].actor_parse_confidence[actor.name] = parse_conf
                previous_actor_responses[actor.name][i] = response

        for critic in critics:
            prompts: list[str] = []
            keys: list[tuple[int, str]] = []
            for actor in actors:
                for i in range(n):
                    prompts.append(
                        build_critic_feedback_prompt(
                            critic,
                            samples[i],
                            dataset_name,
                            round_datas[i].actor_responses[actor.name],
                        )
                    )
                    keys.append((i, actor.name))
            responses = _generate_with_lora(
                inference_engine,
                prompts,
                [critic] * len(prompts),
                lora_requests,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for (sample_idx, actor_name), response in zip(keys, responses):
                round_datas[sample_idx].critic_raw_responses.setdefault(actor_name, {})
                round_datas[sample_idx].critic_raw_responses[actor_name][critic.name] = response

        for i in range(n):
            for actor in actors:
                round_datas[i].critic_feedbacks.setdefault(actor.name, {})
                feedbacks: list[CriticFeedback] = []
                for critic in critics:
                    raw = round_datas[i].critic_raw_responses.get(actor.name, {}).get(critic.name, "")
                    feedback = build_critic_feedback(critic, raw, task_type=task_types[i])
                    round_datas[i].critic_feedbacks[actor.name][critic.name] = feedback
                    feedbacks.append(feedback)
                routed = router.route(feedbacks)
                round_datas[i].routed_feedbacks[actor.name] = routed.selected_feedbacks
                round_datas[i].routed_feedback_texts[actor.name] = routed.feedback_text
                previous_feedback[actor.name][i] = routed.feedback_text

            consensus_answer, consensus_confidence, consensus_weights = _critic_aware_consensus(
                round_datas[i],
                task_types[i],
            )
            round_datas[i].consensus_answer = consensus_answer
            round_datas[i].consensus_confidence = consensus_confidence
            round_datas[i].consensus_weights = consensus_weights
            all_rounds[i].append(round_datas[i])

        logger.info(
            "  Batched round %s/%s (%s samples, %.1fs)",
            round_num + 1,
            num_rounds,
            n,
            time.time() - round_start,
        )

    results = []
    for sample_rounds in all_rounds:
        final_round = sample_rounds[-1]
        results.append(MultiDeliberationResult(
            rounds=sample_rounds,
            final_answers=dict(final_round.actor_answers),
            consensus_answer=final_round.consensus_answer,
            consensus_confidence=final_round.consensus_confidence,
            metadata={"num_actors": len(actors), "num_critics": len(critics)},
        ))
    return results
