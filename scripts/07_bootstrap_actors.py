"""
Bootstrap diverse Actors by generating multi-agent trajectories.

Generates N=5 independent responses per sample with different seeds,
simulates M=2 rounds of debate, and computes consensus via majority vote.

Usage:
    python scripts/07_bootstrap_actors.py \
        --config configs/society/experiment_h100.yaml
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "output_dir": None,
    "num_agents": 5,
    "num_debate_rounds": 2,
    "temperature": 0.8,
    "max_tokens": 1024,
    "seed": 42,
    "max_samples": None,
    "batch_size": 8,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "device": 0,
}


@dataclass
class AgentResponse:
    """Response from a single agent."""
    agent_id: int
    round: int
    response: str
    answer: Optional[str]
    sample_id: str = ""
    response_id: str = ""


@dataclass
class BootstrapTrajectory:
    """Complete trajectory for a single sample."""
    sample_id: str
    sample: dict
    initial_responses: list[AgentResponse]
    debate_rounds: list[list[AgentResponse]]
    consensus_answer: str
    confidence: float
    metadata: dict = field(default_factory=dict)


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Bootstrap diverse Actors",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step01_bootstrap", defaults=STEP_DEFAULTS).to_namespace()


def make_response_id(sample_id: str, round_num: int, agent_id: int) -> str:
    """Stable identifier for a bootstrap response."""
    return f"{sample_id}_round_{round_num}_agent_{agent_id}"


def generate_initial_responses(
    model,
    sample: dict,
    dataset_name: str,
    num_agents: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
    sample_id: str = "sample",
) -> list[AgentResponse]:
    """Generate N independent responses with different seeds."""
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType, append_answer_contract
    from src.algorithms.reward import extract_answer

    prompts = [
        append_answer_contract(
            format_prompt(
                dataset_name,
                PromptType.SINGLE_SHOT,
                sample,
                include_answer_contract=False,
            )
            + f"\n\nYou are bootstrap Agent {agent_id}. Produce an independent solution.",
            sample,
        )
        for agent_id in range(num_agents)
    ]
    random.seed(base_seed)
    gen_results = model.generate(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=base_seed,
    )
    if isinstance(gen_results, str):
        gen_results = [gen_results]

    responses = []
    for agent_id, response_text in enumerate(gen_results):
        response_text = response_text if isinstance(response_text, str) else str(response_text)

        # Extract answer
        answer = extract_answer(response_text, sample.get("task_type", "math"))

        responses.append(AgentResponse(
            agent_id=agent_id,
            round=0,
            response=response_text,
            answer=answer,
            sample_id=sample_id,
            response_id=make_response_id(sample_id, 0, agent_id),
        ))

    return responses


def _coerce_generation_results(results, expected: int, phase: str) -> list[str]:
    """Normalize model output and fail fast on prompt/result misalignment."""
    if isinstance(results, str):
        results = [results]
    results = [r if isinstance(r, str) else str(r) for r in results]
    if len(results) != expected:
        raise ValueError(
            f"{phase} generated {len(results)} responses for {expected} prompts"
        )
    return results


def generate_initial_responses_batch(
    model,
    samples: list[dict],
    dataset_name: str,
    num_agents: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
    sample_ids: Optional[list[str]] = None,
) -> list[list[AgentResponse]]:
    """Generate initial responses for multiple samples and agents in one call."""
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType, append_answer_contract
    from src.algorithms.reward import extract_answer

    prompts: list[str] = []
    meta: list[tuple[int, int]] = []
    sample_ids = sample_ids or [
        str(sample.get("sample_id", f"sample_{sample_idx}"))
        for sample_idx, sample in enumerate(samples)
    ]
    if len(sample_ids) != len(samples):
        raise ValueError("sample_ids and samples must have equal length")
    for sample_idx, sample in enumerate(samples):
        for agent_id in range(num_agents):
            prompts.append(
                append_answer_contract(
                    format_prompt(
                        dataset_name,
                        PromptType.SINGLE_SHOT,
                        sample,
                        include_answer_contract=False,
                    )
                    + (
                        f"\n\nYou are bootstrap Agent {agent_id}. "
                        "Produce an independent solution."
                    ),
                    sample,
                )
            )
            meta.append((sample_idx, agent_id))

    random.seed(base_seed)
    gen_results = model.generate(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=base_seed,
    )
    gen_results = _coerce_generation_results(gen_results, len(prompts), "initial batch")

    responses_by_sample: list[list[AgentResponse]] = [[] for _ in samples]
    for (sample_idx, agent_id), response_text in zip(meta, gen_results):
        sample = samples[sample_idx]
        answer = extract_answer(response_text, sample.get("task_type", "math"))
        responses_by_sample[sample_idx].append(AgentResponse(
            agent_id=agent_id,
            round=0,
            response=response_text,
            answer=answer,
            sample_id=sample_ids[sample_idx],
            response_id=make_response_id(sample_ids[sample_idx], 0, agent_id),
        ))

    return responses_by_sample


def simulate_debate_round(
    model,
    sample: dict,
    dataset_name: str,
    previous_responses: list[AgentResponse],
    round_num: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
    sample_id: str = "sample",
) -> list[AgentResponse]:
    """Simulate one round of debate where agents see others' responses."""
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType, append_answer_contract
    from src.algorithms.reward import extract_answer

    # Format responses text for context
    responses_text = "\n\n".join([
        f"Agent {r.agent_id}: {r.response}"
        for r in previous_responses
    ])

    prompts = [
        format_prompt(
            dataset_name,
            PromptType.DELIBERATION_ACTOR,
            sample,
            include_answer_contract=False,
            responses=responses_text,
        )
        + f"\n\nYou are bootstrap Agent {agent_id}. Revise independently after reading the debate."
        for agent_id in range(len(previous_responses))
    ]
    prompts = [append_answer_contract(prompt, sample) for prompt in prompts]
    seed = base_seed + round_num * 100
    random.seed(seed)
    gen_results = model.generate(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    if isinstance(gen_results, str):
        gen_results = [gen_results]

    responses = []
    for agent_id, response_text in enumerate(gen_results):
        response_text = response_text if isinstance(response_text, str) else str(response_text)

        # Extract answer
        answer = extract_answer(response_text, sample.get("task_type", "math"))

        responses.append(AgentResponse(
            agent_id=agent_id,
            round=round_num,
            response=response_text,
            answer=answer,
            sample_id=sample_id,
            response_id=make_response_id(sample_id, round_num, agent_id),
        ))

    return responses


def simulate_debate_round_batch(
    model,
    samples: list[dict],
    dataset_name: str,
    previous_responses_by_sample: list[list[AgentResponse]],
    round_num: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
    sample_ids: Optional[list[str]] = None,
) -> list[list[AgentResponse]]:
    """Simulate one debate round for multiple samples and agents in one call."""
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType, append_answer_contract
    from src.algorithms.reward import extract_answer

    if len(samples) != len(previous_responses_by_sample):
        raise ValueError("samples and previous responses must have equal length")

    prompts: list[str] = []
    meta: list[tuple[int, int]] = []
    sample_ids = sample_ids or [
        str(sample.get("sample_id", f"sample_{sample_idx}"))
        for sample_idx, sample in enumerate(samples)
    ]
    if len(sample_ids) != len(samples):
        raise ValueError("sample_ids and samples must have equal length")
    for sample_idx, (sample, previous_responses) in enumerate(
        zip(samples, previous_responses_by_sample)
    ):
        responses_text = "\n\n".join([
            f"Agent {r.agent_id}: {r.response}"
            for r in previous_responses
        ])
        for agent_id in range(len(previous_responses)):
            prompts.append(
                format_prompt(
                    dataset_name,
                    PromptType.DELIBERATION_ACTOR,
                    sample,
                    include_answer_contract=False,
                    responses=responses_text,
                )
                + (
                    f"\n\nYou are bootstrap Agent {agent_id}. "
                    "Revise independently after reading the debate."
                )
            )
            prompts[-1] = append_answer_contract(prompts[-1], sample)
            meta.append((sample_idx, agent_id))

    seed = base_seed + round_num * 100
    random.seed(seed)
    gen_results = model.generate(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    gen_results = _coerce_generation_results(
        gen_results, len(prompts), f"debate round {round_num} batch"
    )

    responses_by_sample: list[list[AgentResponse]] = [[] for _ in samples]
    for (sample_idx, agent_id), response_text in zip(meta, gen_results):
        sample = samples[sample_idx]
        answer = extract_answer(response_text, sample.get("task_type", "math"))
        responses_by_sample[sample_idx].append(AgentResponse(
            agent_id=agent_id,
            round=round_num,
            response=response_text,
            answer=answer,
            sample_id=sample_ids[sample_idx],
            response_id=make_response_id(sample_ids[sample_idx], round_num, agent_id),
        ))

    return responses_by_sample


def compute_consensus(responses: list[AgentResponse], task_type: str = "math") -> tuple[str, float]:
    """Compute consensus answer via majority vote with math-aware comparison."""
    from src.algorithms.reward import math_answers_equal

    answers = [r.answer for r in responses if r.answer]

    if not answers:
        return "", 0.0

    if task_type == "math":
        # For math, use math_answers_equal for grouping equivalent answers
        groups: dict[str, list[str]] = {}
        for ans in answers:
            matched_key = None
            for key in groups:
                if math_answers_equal(ans, key):
                    matched_key = key
                    break
            if matched_key:
                groups[matched_key].append(ans)
            else:
                groups[ans] = [ans]
        # Find largest group
        best_key = max(groups, key=lambda k: len(groups[k]))
        confidence = len(groups[best_key]) / len(answers)
        return best_key, confidence
    else:
        counter = Counter(answers)
        most_common = counter.most_common(1)[0]
        consensus = most_common[0]
        confidence = most_common[1] / len(answers)
        return consensus, confidence


def main():
    args = parse_args()

    # Setup directories
    cache_dir = getattr(args, "cache_dir", "output/society") or "output/society"
    output_dir = getattr(args, "output_dir", None) or os.path.join(cache_dir, "bootstrap")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "logs"), exist_ok=True)

    num_agents = args.num_agents
    batch_size = max(1, int(getattr(args, "batch_size", 1) or 1))

    logger.info("=" * 60)
    logger.info("Bootstrap Diverse Actors")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Num agents: {num_agents}")
    logger.info(f"  Debate rounds: {args.num_debate_rounds}")
    logger.info(f"  Bootstrap sample batch size: {batch_size}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("[Step 1] Loading dataset...")
    from src.data.loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    train_data = data.get("train", [])
    test_data = data.get("test", [])

    # Use TRAIN data for bootstrap to avoid data leakage.
    samples = train_data if train_data else test_data

    if args.max_samples:
        samples = samples[:args.max_samples]

    logger.info(f"  Loaded {len(samples)} samples")

    # Load model
    logger.info("[Step 2] Loading model...")
    from src.inference.vllm_server import VLLMInference

    model = VLLMInference(
        args.model_name,
        cuda_device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Generate trajectories
    logger.info("[Step 3] Generating bootstrap trajectories...")
    trajectories = []
    output_file = os.path.join(output_dir, "trajectories.jsonl")

    # Crash recovery: load existing trajectories and skip already-processed samples
    existing_sample_ids = set()
    if os.path.exists(output_file):
        logger.info(f"Found existing {output_file}, resuming from checkpoint...")
        with open(output_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    existing_sample_ids.add(entry["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"  Found {len(existing_sample_ids)} existing trajectories, skipping them")

    pending = [
        (idx, sample)
        for idx, sample in enumerate(samples)
        if f"{args.dataset}_{idx}" not in existing_sample_ids
    ]
    logger.info(f"  Pending samples: {len(pending)}")

    for batch_start in range(0, len(pending), batch_size):
        batch_entries = pending[batch_start:batch_start + batch_size]
        batch_indices = [idx for idx, _ in batch_entries]
        batch_samples = [sample for _, sample in batch_entries]
        batch_sample_ids = [f"{args.dataset}_{idx}" for idx in batch_indices]
        first_display_idx = batch_start + 1
        last_display_idx = batch_start + len(batch_entries)
        logger.info(
            f"  Progress: pending {first_display_idx}-{last_display_idx}/{len(pending)}"
        )

        base_seed = args.seed + batch_indices[0] * 1000

        # Generate initial responses for all samples x agents in this batch.
        initial_responses_batch = generate_initial_responses_batch(
            model,
            batch_samples,
            args.dataset,
            num_agents,
            args.temperature,
            args.max_tokens,
            base_seed,
            sample_ids=batch_sample_ids,
        )

        # Simulate debate rounds, batching all samples x agents per round.
        debate_rounds_batch: list[list[list[AgentResponse]]] = [
            [] for _ in batch_samples
        ]
        current_responses_batch = initial_responses_batch

        for round_num in range(1, args.num_debate_rounds + 1):
            round_responses_batch = simulate_debate_round_batch(
                model,
                batch_samples,
                args.dataset,
                current_responses_batch,
                round_num,
                args.temperature,
                args.max_tokens,
                base_seed,
                sample_ids=batch_sample_ids,
            )
            for local_idx, round_responses in enumerate(round_responses_batch):
                debate_rounds_batch[local_idx].append(round_responses)
            current_responses_batch = round_responses_batch

        batch_records = []
        for local_idx, (idx, sample) in enumerate(batch_entries):
            sample_id = f"{args.dataset}_{idx}"
            initial_responses = initial_responses_batch[local_idx]
            debate_rounds = debate_rounds_batch[local_idx]

            final_responses = debate_rounds[-1] if debate_rounds else initial_responses
            consensus, confidence = compute_consensus(
                final_responses,
                sample.get("task_type", "math"),
            )

            trajectory = BootstrapTrajectory(
                sample_id=sample_id,
                sample=sample,
                initial_responses=[r.__dict__ for r in initial_responses],
                debate_rounds=[[r.__dict__ for r in round_resp] for round_resp in debate_rounds],
                consensus_answer=consensus,
                confidence=confidence,
                metadata={
                    "num_agents": num_agents,
                    "num_rounds": args.num_debate_rounds,
                    "temperature": args.temperature,
                    "batch_size": batch_size,
                },
            )

            trajectories.append(trajectory)
            batch_records.append({
                "sample_id": trajectory.sample_id,
                "sample": trajectory.sample,
                "initial_responses": trajectory.initial_responses,
                "debate_rounds": trajectory.debate_rounds,
                "consensus_answer": trajectory.consensus_answer,
                "confidence": trajectory.confidence,
                "metadata": trajectory.metadata,
            })

        # Write to JSONL after each sample batch for crash recovery.
        with open(output_file, "a") as f:
            for record in batch_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"[Step 4] Saved {len(trajectories)} trajectories to {output_file}")
    logger.info("=" * 60)
    logger.info("Bootstrap complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
