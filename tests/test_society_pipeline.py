"""Tests for society pipeline checkpoint markers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_path = scripts_dir / "13_society_pipeline.py"
    spec = importlib.util.spec_from_file_location("society_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_json_marker_requires_matching_fingerprint(tmp_path):
    pipeline = _load_pipeline_module()
    fingerprint = {
        "config_hash": "config-a",
        "git_commit": "commit-a",
        "input_hash": "input-a",
        "input_paths": ["input.json"],
    }

    pipeline.mark_phase_done(5, str(tmp_path), fingerprint)

    assert pipeline.is_phase_done(5, str(tmp_path), fingerprint)

    changed = dict(fingerprint)
    changed["config_hash"] = "config-b"
    assert not pipeline.is_phase_done(5, str(tmp_path), changed)


def test_legacy_text_marker_is_invalid(tmp_path):
    pipeline = _load_pipeline_module()
    marker = tmp_path / ".phase4_done"
    marker.write_text("completed at 2026-04-30 00:00:00\n")

    assert not pipeline.is_phase_done(4, str(tmp_path), {
        "config_hash": "config",
        "git_commit": "commit",
        "input_hash": "input",
    })


def test_input_hash_changes_with_file_content(tmp_path):
    pipeline = _load_pipeline_module()
    data_file = tmp_path / "classified_data.json"
    data_file.write_text('{"version": 1}\n')
    first = pipeline._hash_paths([str(data_file)])

    data_file.write_text('{"version": 2}\n')
    second = pipeline._hash_paths([str(data_file)])

    assert first != second


def test_strict_classification_phases_inherit_step02_api_key(tmp_path, monkeypatch):
    pipeline = _load_pipeline_module()
    config = tmp_path / "config.yaml"
    config.write_text("""
step02_classify:
  api_key: test-key
""")
    monkeypatch.delenv("GLM_API_KEY", raising=False)

    phase4_env = pipeline._build_subprocess_env(4, str(config))
    phase5_env = pipeline._build_subprocess_env(5, str(config))
    phase3_env = pipeline._build_subprocess_env(3, str(config))

    assert phase4_env["GLM_API_KEY"] == "test-key"
    assert phase5_env["GLM_API_KEY"] == "test-key"
    assert "GLM_API_KEY" not in phase3_env
