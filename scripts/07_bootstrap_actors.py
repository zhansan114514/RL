"""
Bootstrap diverse Actors by generating multi-agent trajectories.

Generates N=5 independent responses per sample with different seeds,
simulates M=2 rounds of debate, and computes consensus via majority vote.

Usage:
    python scripts/07_bootstrap_actors.py \
        --config configs/society/experiment_h100.yaml \
        --max_samples 100 \
        --num_agents 5 \
        --num_debate_rounds 2
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "max_samples", "seed", "cache_dir", "device", "dtype", "gpu_memory_utilization")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "output_dir": None,
    "num_responses": 5,          # Match config key name in experiment_h100.yaml
    "num_agents": 5,             # Alias used internally (fallback)
    "num_debate_rounds": 2,
    "temperature": 0.8,
    "max_tokens": 512,
    "seed": 42,
    "max_samples": None,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "device": 0,
}


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap process."""
    model_name: str
    dataset: str
    num_agents: int
    num_debate_rounds: int
    temperature: float
    max_tokens: int
    seed: int
    max_samples: Optional[int]
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    device: int
    cache_dir: str
    output_dir: Optional[str]


@dataclass
class AgentResponse:
    """Response from a single agent."""
    agent_id: int
    round: int
    response: str
    answer: Optional[str]


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
    return resolve_config(
        cli_args.config, "step01_bootstrap", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )


def generate_initial_responses(
    model,
    sample: dict,
    dataset_name: str,
    num_agents: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
) -> list[AgentResponse]:
    """Generate N independent responses with different seeds."""
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType
    from src.algorithms.reward import extract_answer

    responses = []

    for agent_id in range(num_agents):
        # Use different seed for each agent
        seed = base_seed + agent_id
        random.seed(seed)

        # Format prompt
        prompt = format_prompt(
            dataset_name,
            PromptType.SINGLE_SHOT,
            sample,
        )

        # Generate response
        gen_result = model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        response_text = gen_result[0] if isinstance(gen_result, list) else gen_result

        # Extract answer
        answer = extract_answer(response_text, sample.get("task_type", "math"))

        responses.append(AgentResponse(
            agent_id=agent_id,
            round=0,
            response=response_text,
            answer=answer,
        ))

    return responses


def simulate_debate_round(
    model,
    sample: dict,
    dataset_name: str,
    previous_responses: list[AgentResponse],
    round_num: int,
    temperature: float,
    max_tokens: int,
    base_seed: int,
) -> list[AgentResponse]:
    """Simulate one round of debate where agents see others' responses."""
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType
    from src.algorithms.reward import extract_answer

    responses = []

    # Format responses text for context
    responses_text = "\n\n".join([
        f"Agent {r.agent_id}: {r.response}"
        for r in previous_responses
    ])

    for agent_id in range(len(previous_responses)):
        seed = base_seed + agent_id + round_num * 100

        # Format deliberation prompt
        prompt = format_prompt(
            dataset_name,
            PromptType.DELIBERATION_ACTOR,
            sample,
            responses_text=responses_text,
        )

        # Generate response
        gen_result = model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        response_text = gen_result[0] if isinstance(gen_result, list) else gen_result

        # Extract answer
        answer = extract_answer(response_text, sample.get("task_type", "math"))

        responses.append(AgentResponse(
            agent_id=agent_id,
            round=round_num,
            response=response_text,
            answer=answer,
        ))

    return responses


def compute_consensus(responses: list[AgentResponse], task_type: str = "math") -> tuple[str, float]:
    """Compute consensus answer via majority vote with math-aware comparison."""
    from src.algorithms.reward import math_answers_equal, normalize_answer

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

    # num_responses from config takes priority over num_agents
    num_agents = getattr(args, "num_responses", None) or getattr(args, "num_agents", 5)

    logger.info("=" * 60)
    logger.info("Bootstrap Diverse Actors")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Num agents: {num_agents}")
    logger.info(f"  Debate rounds: {args.num_debate_rounds}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("[Step 1] Loading dataset...")
    from src.data.loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    train_data = data.get("train", [])
    test_data = data.get("test", [])

    # Use TRAIN data for bootstrap to avoid data leakage.
    # Bootstrap trajectories are used to build DPO preference pairs for
    # Actor/Critic diversification (scripts 09/10), so they must NOT
    # contain test data that will later be used for evaluation.
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

    for idx, sample in enumerate(samples):
        if (idx + 1) % 10 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(samples)}")

        sample_id = f"{args.dataset}_{idx}"

        # Generate initial responses
        initial_responses = generate_initial_responses(
            model,
            sample,
            args.dataset,
            num_agents,
            args.temperature,
            args.max_tokens,
            args.seed + idx * 1000,
        )

        # Simulate debate rounds
        debate_rounds = []
        current_responses = initial_responses

        for round_num in range(1, args.num_debate_rounds + 1):
            round_responses = simulate_debate_round(
                model,
                sample,
                args.dataset,
                current_responses,
                round_num,
                args.temperature,
                args.max_tokens,
                args.seed + idx * 1000,
            )
            debate_rounds.append(round_responses)
            current_responses = round_responses

        # Compute consensus from final round
        final_responses = debate_rounds[-1] if debate_rounds else initial_responses
        consensus, confidence = compute_consensus(final_responses, sample.get("task_type", "math"))

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
            },
        )

        trajectories.append(trajectory)

        # Write to JSONL incrementally
        with open(output_file, "a") as f:
            f.write(json.dumps({
                "sample_id": trajectory.sample_id,
                "sample": trajectory.sample,
                "initial_responses": trajectory.initial_responses,
                "debate_rounds": trajectory.debate_rounds,
                "consensus_answer": trajectory.consensus_answer,
                "confidence": trajectory.confidence,
                "metadata": trajectory.metadata,
            }, ensure_ascii=False) + "\n")

    logger.info(f"[Step 4] Saved {len(trajectories)} trajectories to {output_file}")
    logger.info("=" * 60)
    logger.info("Bootstrap complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
