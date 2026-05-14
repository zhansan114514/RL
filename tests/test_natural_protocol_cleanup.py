"""Regression tests for removing strict-format training/evaluation surfaces."""

from __future__ import annotations

import importlib.util
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


def test_society_configs_do_not_expose_schema_valid_router_flags():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs" / "society").rglob("*.yaml"):
        text = path.read_text()
        assert "router_require_schema_valid" not in text
        assert "require_schema_valid" not in text


def test_training_pair_mixes_do_not_include_format_bucket():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs" / "society").rglob("*.yaml"):
        text = path.read_text()
        assert "\n    format:" not in text


def test_evaluation_payload_uses_parsing_diagnostics_not_format_metrics():
    evaluate = _load_script("12_society_evaluate.py")
    payload = evaluate._build_results_payload({})

    assert "parsing_diagnostics" in payload
    assert "format_metrics" not in payload


def test_actor_training_quality_gate_rejects_answer_only_outputs():
    from src.society.actor_response_quality import is_trainable_actor_response

    assert not is_trainable_actor_response("A", "multiple_choice")
    assert not is_trainable_actor_response("The final result is A.", "multiple_choice")
    assert is_trainable_actor_response(
        "The question states the key evidence, so option A follows.\n"
        "The final result is A.",
        "multiple_choice",
    )
