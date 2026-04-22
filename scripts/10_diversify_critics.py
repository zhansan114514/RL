"""
Diversify Critics by training specialized LoRA adapters for each error type.

For each Critic:
1. Load base model
2. Filter classified data by error type
3. Build DPO preference pairs (chosen=specialty-guided feedback, rejected=generic feedback)
4. Train with DPO
5. Save to cache/society/critics/{agent_id}/

Usage:
    python scripts/10_diversify_critics.py \
        --config configs/society/experiment_h100.yaml \
        --error_types arithmetic logic hallucination verification
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "input_dir", "output_dir", "seed", "device", "dtype", "gpu_memory_utilization")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "input_dir": "output/society/classified",
    "actor_dir": "output/society/actors",
    "output_dir": "output/society/critics",
    "error_types": ["arithmetic", "logic", "hallucination", "verification"],
    "max_delib_samples": 50,
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "beta": 0.1,
    "num_rounds": 5,
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Diversify Critics",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return resolve_config(
        cli_args.config, "step04_diversify_critics", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )


def load_classified_data(input_dir: str) -> Dict[str, Any]:
    """Load classified data."""
    classified_file = os.path.join(input_dir, "classified_data.json")

    if not os.path.exists(classified_file):
        raise FileNotFoundError(f"Classified data not found: {classified_file}")

    with open(classified_file) as f:
        data = json.load(f)

    return data


def build_critic_preference_pairs(
    classified_results: List[Dict],
    trajectories: Dict[str, Any],
    error_type: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: int = 0,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.65,
    max_model_len: int = 4096,
    engine=None,
) -> List[Dict[str, Any]]:
    """
    Build DPO preference pairs for Critic training using LLM-generated feedback.

    Uses the LLM to generate:
    - chosen: Guided feedback specific to the error type
    - rejected: Generic/vague feedback

    This avoids hardcoded template strings and produces real LLM feedback.
    """
    from src.algorithms.reward import extract_answer, math_answers_equal

    preference_pairs = []

    # Filter samples with this error type (core of data-level diversification)
    error_samples = [
        r for r in classified_results
        if r.get("error_type") == error_type
    ]

    logger.info(f"  Found {len(error_samples)} samples for error type '{error_type}'")

    # Only use error-type-filtered samples so each Critic specializes on its error domain
    # Fallback to untyped samples when no explicit labels exist (not to ALL samples)
    if not error_samples:
        untyped = [r for r in classified_results if not r.get("error_type")]
        if untyped:
            logger.warning(
                f"  No typed samples for '{error_type}', "
                f"using {len(untyped)} untyped samples as fallback"
            )
            error_samples = untyped
        else:
            logger.warning(f"  No samples available for error type '{error_type}'")
            return []

    # Collect candidate responses for LLM feedback generation
    candidates = []
    for result in error_samples:
        sample_id = result["sample_id"]

        if sample_id not in trajectories:
            continue

        traj = trajectories[sample_id]
        sample = traj["sample"]
        correct_answer = sample.get("answer", "")
        question = sample.get("question", "")

        # Collect all responses across rounds
        all_responses = list(traj.get("initial_responses", []))
        for round_responses in traj.get("debate_rounds", []):
            all_responses.extend(round_responses)

        # Find wrong answers as context for critic training
        wrong_responses = []
        correct_responses = []
        for resp in all_responses:
            response_text = resp.get("response", "")
            answer = resp.get("answer")
            task_type = sample.get("task_type", "math")

            # Use math_answers_equal for math tasks
            if task_type == "math":
                is_correct = math_answers_equal(answer or "", correct_answer)
            else:
                is_correct = (answer or "").upper() == (correct_answer or "").upper()

            if not is_correct:
                wrong_responses.append(response_text)
            else:
                correct_responses.append(response_text)

        if wrong_responses:
            candidates.append({
                "sample": sample,
                "question": question,
                "wrong_response": wrong_responses[0],
                "correct_answer": correct_answer,
            })

    if not candidates:
        logger.warning(f"  No wrong-answer candidates found for {error_type}, skipping")
        return []

    # Use provided engine or create a new one
    from src.inference.vllm_server import VLLMInference

    engine_provided = engine is not None
    try:
        if engine is None:
            engine = VLLMInference(
                model_name,
                cuda_device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )

        for cand in candidates:
            question = cand["question"]
            wrong_resp = cand["wrong_response"]
            correct_answer = cand["correct_answer"]

            # Chosen: guided feedback with error-type specificity
            guided_prompt = (
                f"You are reviewing a solution and have identified a {error_type} error.\n"
                f"Problem: {question}\n"
                f"Student's solution: {wrong_resp}\n"
                f"Correct answer: {correct_answer}\n\n"
                f"Provide specific, actionable feedback that identifies the {error_type} error "
                f"and guides toward the correct solution. "
                f"After your analysis, output your confidence on a scale of 0.0 to 1.0 "
                f"using the format: [Confidence: 0.X]"
            )
            chosen = engine.generate_single(guided_prompt, max_tokens=256, temperature=0.3)

            # Rejected: generic feedback without specialty guidance
            generic_prompt = (
                f"Review this solution briefly.\n"
                f"Problem: {question}\n"
                f"Solution: {wrong_resp}\n\n"
                f"Provide brief feedback. "
                f"After your analysis, output your confidence on a scale of 0.0 to 1.0 "
                f"using the format: [Confidence: 0.X]"
            )
            rejected = engine.generate_single(generic_prompt, max_tokens=256, temperature=0.7)

            if chosen and rejected and chosen.strip() != rejected.strip():
                preference_pairs.append({
                    "sample": cand["sample"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "error_type": error_type,
                        "sample_id": cand["sample"].get("sample_id", ""),
                        "had_wrong_response": True,
                    },
                })

        if not engine_provided:
            del engine
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"  Failed to generate LLM feedback: {e}")

    logger.info(f"  Built {len(preference_pairs)} preference pairs")
    return preference_pairs


def train_critic_dpo(
    model_name: str,
    preference_pairs: List[Dict],
    error_type: str,
    output_dir: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    max_length: int,
    beta: float,
    seed: int,
    device: int,
) -> str:
    """Train Critic with DPO."""
    from datasets import Dataset
    from src.training.dpo_trainer import train_dpo
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)

    # Create output directory
    critic_output_dir = os.path.join(output_dir, f"critic_{error_type}")
    os.makedirs(critic_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{error_type}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {critic_output_dir}")

    # Convert preference_pairs to HuggingFace Dataset
    hf_data = {
        "prompt": [p["sample"].get("question", "") for p in preference_pairs],
        "chosen": [p["chosen"] for p in preference_pairs],
        "rejected": [p["rejected"] for p in preference_pairs],
    }
    preference_dataset = Dataset.from_dict(hf_data)

    # Train DPO
    checkpoint_path = train_dpo(
        model_name_or_path=model_name,
        preference_dataset=preference_dataset,
        output_dir=critic_output_dir,
        model_type=model_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        max_length=max_length,
        beta=beta,
        seed=seed,
        device=device,
    )

    logger.info(f"  Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def main():
    args = parse_args()

    # Setup directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diversify Critics")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Actor dir: {args.actor_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Error types: {args.error_types}")
    logger.info("=" * 60)

    # Load classified data
    logger.info("[Step 1] Loading classified data...")
    classified_data = load_classified_data(input_dir)
    classified_results = classified_data["results"]

    # Load trajectories
    logger.info("[Step 2] Loading bootstrap trajectories...")

    # Load trajectories directly
    bootstrap_dir = os.path.join(args.cache_dir, "bootstrap")
    trajectory_file = os.path.join(bootstrap_dir, "trajectories.jsonl")

    trajectories = {}
    if os.path.exists(trajectory_file):
        with open(trajectory_file) as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    trajectories[traj["sample_id"]] = traj
        logger.info(f"  Loaded {len(trajectories)} trajectories")

    # Train each critic
    logger.info("[Step 3] Training specialized critics...")

    critic_paths = {}

    # Create a single vLLM engine to reuse across all error types
    from src.inference.vllm_server import VLLMInference
    shared_engine = None
    try:
        shared_engine = VLLMInference(
            args.model_name,
            cuda_device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=4096,
        )
    except Exception as e:
        logger.warning(f"Failed to create shared vLLM engine, will create per-error-type: {e}")

    try:
        # Phase 1: Use shared engine to generate preference pairs for ALL error types
        all_pairs = {}
        for error_type in args.error_types:
            logger.info(f"\n--- Building pairs for Critic: {error_type} ---")

            preference_pairs = build_critic_preference_pairs(
                classified_results,
                trajectories,
                error_type,
                model_name=args.model_name,
                device=args.device,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                engine=shared_engine,
            )

            if preference_pairs:
                all_pairs[error_type] = preference_pairs
            else:
                logger.warning(f"  No preference pairs for '{error_type}', skipping")

        # Clean up shared engine before DPO training (GPU memory intensive)
        if shared_engine is not None:
            del shared_engine
            shared_engine = None
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Phase 2: Train each critic (no engine needed, DPO runs in subprocess)
        for error_type, preference_pairs in all_pairs.items():
            logger.info(f"\n--- Training Critic: {error_type} ---")

            checkpoint_path = train_critic_dpo(
                model_name=args.model_name,
                preference_pairs=preference_pairs,
                error_type=error_type,
                output_dir=output_dir,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_epochs=args.num_epochs,
                max_length=args.max_length,
                beta=args.beta,
                seed=args.seed,
                device=args.device,
            )

            critic_paths[error_type] = checkpoint_path

    finally:
        # Clean up shared engine if still alive (e.g. on early exit)
        if shared_engine is not None:
            del shared_engine
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Save critic registry
    logger.info("\n[Step 4] Saving critic registry...")

    registry_file = os.path.join(output_dir, "critic_registry.json")
    with open(registry_file, "w") as f:
        json.dump({
            "critics": {
                error_type: {
                    "error_type": error_type,
                    "model_path": path,
                    "base_model": args.model_name,
                }
                for error_type, path in critic_paths.items()
            },
            "metadata": {
                "base_model": args.model_name,
                "num_critics": len(critic_paths),
            },
        }, f, indent=2)

    logger.info(f"  Registry saved: {registry_file}")

    logger.info("=" * 60)
    logger.info("Critic diversification complete!")
    logger.info(f"  Trained {len(critic_paths)} critics:")
    for error_type, path in critic_paths.items():
        logger.info(f"    {error_type}: {path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
