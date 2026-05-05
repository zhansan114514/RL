"""
Debug: Run A5 deliberation on a few wrong-answer samples and print full traces.
Shows actor responses, critic feedbacks, router output for each round.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "mmlu",
    "seed": 42,
    "evaluation_mode": "single_gpu",
    "evaluation_modes": {
        "single_gpu": {
            "devices": [6],
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.60,
        },
        "dual_gpu": {
            "devices": [6, 7],
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.80,
        },
    },
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "society_dir": "output/society_mmlu/society",
    "output_dir": "output/society_mmlu/eval",
    "num_rounds": 2,
    "max_tokens": 512,
    "temperature": 0.7,
    "router_top_k": 2,
    "sampling": None,
    "mmlu_load_mode": "by_subject",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--num_debug", type=int, default=3, help="Number of wrong samples to trace")
    cli_args = parser.parse_args()

    from src.utils.config import ConfigManager
    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step06_evaluate", defaults=STEP_DEFAULTS).to_namespace()

    # Load registry
    registry_path = os.path.join(args.society_dir, "final_agent_registry.json")
    with open(registry_path) as f:
        registry = json.load(f)

    base_model = registry.get("training_config", {}).get("base_model", args.model_name)
    all_actor_names = list(registry["actors"].keys())
    all_critic_names = list(registry["critics"].keys())

    # Load dataset
    from src.data.loader import load_dataset as load_data
    data = load_data(
        args.dataset, seed=args.seed,
        sampling=getattr(args, "sampling", None),
        mmlu_load_mode=getattr(args, "mmlu_load_mode", "by_subject"),
    )
    samples = data.get("test", [])

    if not samples:
        raise ValueError(f"Test split is empty for dataset={args.dataset}")

    # First, run all samples to find wrong ones
    from importlib import import_module
    eval_module = import_module("scripts.12_society_evaluate")
    _build_agent_configs = eval_module._build_agent_configs
    runtime = eval_module.resolve_evaluation_runtime(args)

    from src.inference.vllm_server import VLLMInference
    total_lora = len(all_actor_names) + len(all_critic_names)

    logger.info(
        f"Loading engine with evaluation mode {runtime.mode} "
        f"(devices={runtime.devices}, "
        f"tensor_parallel_size={runtime.tensor_parallel_size})..."
    )
    engine = VLLMInference(
        base_model,
        cuda_device=runtime.devices,
        tensor_parallel_size=runtime.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=runtime.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_loras=total_lora,
        max_lora_rank=256,
    )

    try:
        a_configs, c_configs, lora = _build_agent_configs(
            registry,
            actor_names=all_actor_names[:3],
            critic_names=all_critic_names,
        )

        from src.society.multi_deliberation import multi_agent_deliberate_single_gpu
        from src.society.router import CriticRouter
        from collections import Counter

        router = CriticRouter(top_k=args.router_top_k, uniform_weights=False)

        # Run on all samples, collect wrong ones
        wrong_samples = []
        logger.info(f"Running all {len(samples)} samples to find errors...")
        for si, sample in enumerate(samples):
            result = multi_agent_deliberate_single_gpu(
                inference_engine=engine,
                actors=a_configs,
                critics=c_configs,
                sample=sample,
                dataset_name=args.dataset,
                lora_paths=lora,
                num_rounds=args.num_rounds,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                router=router,
            )

            # initial answer
            first_round = result.rounds[0]
            init_answers = [a for a in first_round.actor_answers.values() if a is not None]
            initial_answer = Counter(init_answers).most_common(1)[0][0] if init_answers else ""
            final_answer = result.consensus_answer or ""

            ground_truth = sample.get("answer", "")
            init_correct = (initial_answer == ground_truth)
            final_correct = (final_answer == ground_truth)

            # Only collect cases where initial was correct but final became wrong
            if init_correct and not final_correct:
                wrong_samples.append((si, sample, result, initial_answer, final_answer, ground_truth))
                logger.info(f"  Sample {si}: initial={initial_answer}(correct) -> final={final_answer}(wrong), truth={ground_truth}")
                if len(wrong_samples) >= cli_args.num_debug:
                    break

        if not wrong_samples:
            logger.info("No samples where correct->wrong found. Showing all wrong final answers...")
            # Try any wrong answer
            for si, sample in enumerate(samples):
                result = multi_agent_deliberate_single_gpu(
                    inference_engine=engine,
                    actors=a_configs,
                    critics=c_configs,
                    sample=sample,
                    dataset_name=args.dataset,
                    lora_paths=lora,
                    num_rounds=args.num_rounds,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    router=router,
                )
                final_answer = result.consensus_answer or ""
                ground_truth = sample.get("answer", "")
                if final_answer != ground_truth:
                    first_round = result.rounds[0]
                    init_answers = [a for a in first_round.actor_answers.values() if a is not None]
                    initial_answer = Counter(init_answers).most_common(1)[0][0] if init_answers else ""
                    wrong_samples.append((si, sample, result, initial_answer, final_answer, ground_truth))
                    logger.info(f"  Sample {si}: initial={initial_answer} -> final={final_answer}(wrong), truth={ground_truth}")
                    if len(wrong_samples) >= cli_args.num_debug:
                        break

        # Print detailed traces
        print("\n" + "=" * 80)
        print("DETAILED DELIBERATION TRACES")
        print("=" * 80)

        for idx, (si, sample, result, initial_answer, final_answer, ground_truth) in enumerate(wrong_samples):
            print(f"\n{'#' * 80}")
            print(f"### Sample {si} (case {idx+1}/{len(wrong_samples)})")
            print(f"{'#' * 80}")
            print(f"Question: {sample.get('question', '')[:200]}")
            if sample.get("choices"):
                for ci, c in enumerate(sample["choices"]):
                    label = chr(65 + ci)  # A, B, C, D
                    print(f"  {label}. {c}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Initial Answer (R0): {initial_answer} {'✓' if initial_answer == ground_truth else '✗'}")
            print(f"Final Answer (R{len(result.rounds)-1}): {final_answer} {'✓' if final_answer == ground_truth else '✗'}")
            print()

            for rnd in result.rounds:
                print(f"  --- Round {rnd.round_num} ---")

                # Actor responses
                for actor_name, resp in rnd.actor_responses.items():
                    answer = rnd.actor_answers.get(actor_name, "?")
                    print(f"  [{actor_name}] answer={answer}")
                    # Print last 200 chars of response
                    print(f"    Response (tail): ...{resp[-300:]}")
                    print()

                # Critic feedbacks
                for actor_name, critic_dict in rnd.critic_feedbacks.items():
                    print(f"  Critic feedbacks for {actor_name}:")
                    for critic_name, fb in critic_dict.items():
                        print(f"    [{critic_name}] {fb[:300]}")
                    print()

                # Routed feedback
                for actor_name, routed in rnd.routed_feedbacks.items():
                    if routed:
                        print(f"  Routed feedback for {actor_name}:")
                        print(f"    {routed.feedback_text[:400]}")
                        print()

                # Consensus
                print(f"  Consensus: {rnd.consensus_answer}")
                print()

        # Save traces
        traces = []
        for si, sample, result, initial_answer, final_answer, ground_truth in wrong_samples:
            trace = {
                "sample_idx": si,
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "ground_truth": ground_truth,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "rounds": [],
            }
            for rnd in result.rounds:
                round_info = {
                    "round_num": rnd.round_num,
                    "actor_answers": dict(rnd.actor_answers),
                    "actor_responses": {k: v[-500:] for k, v in rnd.actor_responses.items()},
                    "critic_feedbacks": {a: {c: fb[:500] for c, fb in cdict.items()} for a, cdict in rnd.critic_feedbacks.items()},
                    "routed_feedbacks": {a: rf.feedback_text[:500] if rf else "" for a, rf in rnd.routed_feedbacks.items()},
                    "consensus_answer": rnd.consensus_answer,
                }
                trace["rounds"].append(round_info)
            traces.append(trace)

        out_path = os.path.join(args.output_dir, "debug_traces.json")
        with open(out_path, "w") as f:
            json.dump(traces, f, indent=2, ensure_ascii=False)
        logger.info(f"Traces saved to {out_path}")

    finally:
        del engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
