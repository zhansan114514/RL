"""Tests for configuration management."""

import pytest
from pathlib import Path

from src.utils.config import ConfigManager, StepConfig, ConfigKeyError


@pytest.fixture(autouse=True)
def reset_config():
    """Reset ConfigManager singleton before and after each test."""
    ConfigManager.reset()
    yield
    ConfigManager.reset()


class TestConfigManagerInitialize:
    """Test ConfigManager initialization and singleton behavior."""

    def test_initialize_defaults_only(self):
        """Loading with no arguments should load only default.yaml."""
        cfg = ConfigManager.initialize()
        assert cfg.get("seed") == 42
        assert cfg.get("inference.max_model_len") == 4096
        assert cfg.get("training.lora_r") == 256

    def test_initialize_with_experiment_config(self):
        """Loading with experiment config should merge common section."""
        cfg = ConfigManager.initialize(
            config_path="configs/society/experiment_mmlu.yaml",
            load_local=False,
        )
        # common section overrides defaults
        assert cfg.get("seed") == 42

    def test_initialize_singleton(self):
        """Second initialize() should reinitialize the singleton."""
        cfg1 = ConfigManager.initialize()
        cfg2 = ConfigManager.instance()
        assert cfg1 is cfg2

    def test_initialize_reset(self):
        """reset() should clear the singleton."""
        ConfigManager.initialize()
        assert ConfigManager.is_initialized()
        ConfigManager.reset()
        assert not ConfigManager.is_initialized()

    def test_instance_without_initialize_raises(self):
        """instance() should raise if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            ConfigManager.instance()

    def test_initialize_with_overrides(self):
        """CLI-style overrides should take precedence."""
        cfg = ConfigManager.initialize(
            overrides=["inference.max_model_len=8192"],
        )
        assert cfg.get("inference.max_model_len") == 8192


class TestConfigManagerGet:
    """Test get/require/section methods."""

    def test_get_existing_key(self):
        cfg = ConfigManager.initialize()
        assert cfg.get("seed") == 42

    def test_get_missing_key_returns_default(self):
        cfg = ConfigManager.initialize()
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_missing_key_no_default(self):
        cfg = ConfigManager.initialize()
        assert cfg.get("nonexistent.key") is None

    def test_require_existing_key(self):
        cfg = ConfigManager.initialize()
        assert cfg.require("seed") == 42

    def test_require_missing_key_raises(self):
        cfg = ConfigManager.initialize()
        with pytest.raises(ConfigKeyError, match="nonexistent"):
            cfg.require("nonexistent.key")

    def test_section(self):
        cfg = ConfigManager.initialize()
        inference = cfg.section("inference")
        assert isinstance(inference, dict)
        assert "max_model_len" in inference

    def test_section_missing_returns_empty(self):
        cfg = ConfigManager.initialize()
        assert cfg.section("nonexistent") == {}


class TestConfigManagerStep:
    """Test step() method for step-specific config merging."""

    def test_step_merge_priority(self, tmp_path):
        """Step section should override common, and defaults act as required keys."""
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("""
common:
  model_name: Qwen/Qwen2.5-7B-Instruct
  num_agents: 3
step01_bootstrap:
  num_agents: 5
  temperature: 0.8
""")

        cfg = ConfigManager.initialize(
            config_path=str(experiment),
            load_local=False,
        )
        step = cfg.step("step01_bootstrap", defaults={
            "num_agents": True,
            "temperature": True,
        })
        assert step.get("num_agents") == 5       # from step section
        assert step.get("temperature") == 0.8     # from step section
        assert step.get("model_name") == "Qwen/Qwen2.5-7B-Instruct"  # from common

    def test_step_defaults_are_required_schema(self):
        """defaults declares required keys; it does not fill missing values."""
        cfg = ConfigManager.initialize()
        with pytest.raises(ConfigKeyError, match="count, foo"):
            cfg.step("step01", defaults={"foo": "bar", "count": 10})

    def test_step_missing_section_returns_common_when_no_required_keys(self, tmp_path):
        """Missing step section should still return common config."""
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("""
common:
  model_name: Qwen/Qwen2.5-7B-Instruct
""")
        cfg = ConfigManager.initialize(
            config_path=str(experiment),
            load_local=False,
        )
        step = cfg.step("step99_nonexistent")
        assert step.get("model_name") == "Qwen/Qwen2.5-7B-Instruct"  # from common

    def test_step_none_values_do_not_override_common(self, tmp_path):
        """None values in step sections should not erase existing common values."""
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("""
common:
  key: common_val
step01:
  key:
""")
        cfg = ConfigManager.initialize(config_path=str(experiment), load_local=False)
        step = cfg.step("step01", defaults={"key": True})
        assert step.get("key") == "common_val"

    def test_step_unknown_api_provider_raises(self, tmp_path):
        """A selected provider must exist in api.providers."""
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("""
api:
  provider: glm
  providers:
    gpt:
      api_key: secret
      api_base: https://api.example.test
      api_model: gpt-test
step02_classify:
  output_dir: out
""")
        cfg = ConfigManager.initialize(config_path=str(experiment), load_local=False)

        with pytest.raises(ConfigKeyError, match="API provider 'glm' is not defined"):
            cfg.step("step02_classify")

    def test_step_inherits_api_from_local_config(self, tmp_path):
        """Local config should provide API secrets without tracked experiment YAML."""
        experiment = tmp_path / "experiment.yaml"
        local = tmp_path / "local.yaml"
        experiment.write_text("""
common:
  model_name: test-model
step02_classify:
  api_base: https://api.example.test
""")
        local.write_text("""
api:
  api_key: local-secret
  api_model: local-model
""")

        cfg = ConfigManager.initialize(
            config_path=str(experiment),
            local_config_path=str(local),
        )
        step = cfg.step("step02_classify")

        assert step.get("api_key") == "local-secret"
        assert step.get("api_base") == "https://api.example.test"
        assert step.get("api_model") == "local-model"
        assert Path(str(local)).resolve().as_posix() in cfg.loaded_paths

    def test_step_config_overrides_top_level_api(self, tmp_path):
        """Step API values should take precedence over top-level API defaults."""
        experiment = tmp_path / "experiment.yaml"
        experiment.write_text("""
api:
  api_base: https://api.default.test
  api_model: default-model
step02_classify:
  api_base: https://api.step.test
""")

        cfg = ConfigManager.initialize(config_path=str(experiment), load_local=False)
        step = cfg.step("step02_classify")

        assert step.get("api_base") == "https://api.step.test"
        assert step.get("api_model") == "default-model"


class TestStepConfig:
    """Test StepConfig class."""

    def test_attribute_access(self):
        sc = StepConfig({"model_name": "test-model", "seed": 42})
        assert sc.model_name == "test-model"
        assert sc.seed == 42

    def test_attribute_missing_raises(self):
        sc = StepConfig({"model_name": "test"})
        with pytest.raises(AttributeError, match="nonexistent"):
            sc.nonexistent

    def test_get_method(self):
        sc = StepConfig({"key": "value"})
        assert sc.get("key") == "value"
        assert sc.get("missing", "default") == "default"

    def test_to_namespace(self):
        sc = StepConfig({"model_name": "test", "seed": 42})
        ns = sc.to_namespace()
        assert ns.model_name == "test"
        assert ns.seed == 42

    def test_to_dict(self):
        data = {"model_name": "test", "seed": 42}
        sc = StepConfig(data)
        assert sc.to_dict() == data

    def test_contains(self):
        sc = StepConfig({"key": "value"})
        assert "key" in sc
        assert "missing" not in sc


class TestConfigManagerWrite:
    """Test set/merge/save methods."""

    def test_set_override(self):
        cfg = ConfigManager.initialize()
        cfg.set("seed", 123)
        assert cfg.get("seed") == 123

    def test_merge_override(self):
        cfg = ConfigManager.initialize()
        cfg.merge({"seed": 99, "inference": {"max_model_len": 8192}})
        assert cfg.get("seed") == 99
        assert cfg.get("inference.max_model_len") == 8192

    def test_to_dict(self):
        cfg = ConfigManager.initialize()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["seed"] == 42


class TestModelManagerIntegration:
    """Test that model_manager's _get_config still works."""

    def test_get_config_with_initialized_manager(self):
        from src.training.model_manager import _get_config

        ConfigManager.initialize(config_path="configs/society/experiment_mmlu.yaml", load_local=False)
        assert _get_config("inference.gpu_memory_utilization", 0.45) == 0.45
        assert _get_config("inference.max_model_len", 4096) == 4096

    def test_get_config_without_initialized_manager(self):
        from src.training.model_manager import _get_config

        assert _get_config("any.key", "fallback") == "fallback"
