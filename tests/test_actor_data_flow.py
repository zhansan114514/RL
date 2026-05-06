"""Tests for style-prompted classification and actor pair construction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_path = scripts_dir / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classification_checkpoint_metadata_tracks_input_fingerprint(tmp_path):
    classify = _load_script("08_classify_data.py")
    input_dir = tmp_path / "bootstrap"
    input_dir.mkdir()
    trajectory_file = input_dir / "trajectories.jsonl"
    trajectory_file.write_text('{"sample_id":"mmlu_0"}\n')
    args = type("Args", (), {
        "input_dir": str(input_dir),
        "min_style_confidence": 0.65,
    })()

    first = classify.checkpoint_metadata(args)
    trajectory_file.write_text('{"sample_id":"mmlu_0","changed":true}\n')
    second = classify.checkpoint_metadata(args)

    assert first["input_file"]["sha256"] != second["input_file"]["sha256"]


def test_actor_pairs_use_only_accepted_target_style_chosen():
    diversify = _load_script("09_diversify_actors.py")
    sample = {
        "question": "Q",
        "answer": "A",
        "task_type": "multiple_choice",
    }
    classified_results = [{
        "sample_id": "mmlu_0",
        "sample": sample,
        "per_response_labels": [
            {
                "response_id": "chosen",
                "response": "FINAL_ANSWER: A\nRATIONALE:\nLet x = 1.",
                "answer": "A",
                "is_correct": True,
                "primary_style": "algebraic",
                "reasoning_style_confidence": 0.9,
                "format_status": "valid",
                "prompted_style": "algebraic",
                "style_match": True,
                "accepted_for_actor": True,
            },
            {
                "response_id": "wrong",
                "response": "FINAL_ANSWER: B\nRATIONALE:\nWrong.",
                "answer": "B",
                "is_correct": False,
                "format_status": "valid",
                "prompted_style": "algebraic",
                "accepted_for_actor": False,
            },
            {
                "response_id": "other_style",
                "response": "FINAL_ANSWER: A\nRATIONALE:\nDirect.",
                "answer": "A",
                "is_correct": True,
                "primary_style": "direct",
                "reasoning_style_confidence": 0.8,
                "format_status": "valid",
                "prompted_style": "direct",
                "style_match": True,
                "accepted_for_actor": True,
            },
            {
                "response_id": "answer_only",
                "response": "FINAL_ANSWER: A",
                "answer": "A",
                "is_correct": True,
                "primary_style": "algebraic",
                "reasoning_style_confidence": 0.95,
                "format_status": "answer_only",
                "prompted_style": "algebraic",
                "style_match": True,
                "accepted_for_actor": False,
            },
        ],
    }]
    args = type("Args", (), {
        "pair_mix": {"correctness": 0.34, "style": 0.33, "format": 0.33},
        "target_pairs_per_actor": 3,
        "max_pairs_per_actor": 10,
        "max_pairs_per_sample": 10,
    })()

    by_sample = diversify.build_response_index(classified_results)
    pairs, metrics = diversify.build_preference_pairs_for_style(
        by_sample,
        "algebraic",
        args,
    )

    assert {p["metadata"]["pair_type"] for p in pairs} == {
        "correctness",
        "style",
        "format",
    }
    assert all(p["metadata"]["chosen_response_id"] == "chosen" for p in pairs)
    assert metrics["candidate_counts"] == {
        "correctness": 1,
        "style": 2,
        "format": 1,
    }
