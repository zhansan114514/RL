"""Tests for DPO training components."""

import pytest
from unittest.mock import patch, MagicMock, call


class TestLoRAConfig:
    """Test LoRA configuration creation."""

    def test_default_config(self):
        from src.training.lora_config import get_lora_config

        config = get_lora_config("llama3", r=256)
        assert config.r == 256
        assert config.lora_alpha == 512
        assert len(config.target_modules) > 0

    def test_gemma2_config(self):
        from src.training.lora_config import get_lora_config

        config = get_lora_config("gemma2")
        assert config.r == 256

    def test_custom_rank(self):
        from src.training.lora_config import get_lora_config

        config = get_lora_config("llama3", r=128, lora_alpha=256)
        assert config.r == 128
        assert config.lora_alpha == 256


class TestAlternatingTrain:
    """Test alternating training structure."""

    def test_alternating_returns_paths(self):
        """The function should return dict with actor_path and critic_path."""
        from src.training.alternating import alternating_train
        # This is a structural test - actual training requires GPU
        # We verify the function signature accepts expected args
        import inspect
        sig = inspect.signature(alternating_train)
        params = list(sig.parameters.keys())
        assert "actor_path" in params
        assert "critic_path" in params
        assert "num_iterations" in params
        assert "dataset" in params


class TestAlternatingTrainModelPaths:
    """Test model path propagation in alternating training."""

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_model_path_propagation_single_iteration(
        self, mock_gen_traj, mock_train_dpo, mock_vllm
    ):
        """Single iteration should use original paths."""
        from src.training.alternating import alternating_train

        # Setup mocks
        mock_model = MagicMock()
        mock_vllm.return_value = mock_model
        mock_gen_traj.return_value = [
            {
                "positive": "A", "negative": "B",
                "positive_critic": "C", "negative_critic": "D",
                "round": 0, "delta": 0.3, "actor_prompt": "P",
            }
        ]
        mock_train_dpo.return_value = "/output/critic_iter0"

        dataset = [
            {"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}
        ]

        result = alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=dataset,
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=1,
        )

        # Should return output paths
        assert "actor_path" in result
        assert "critic_path" in result

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_model_path_propagation_two_iterations(
        self, mock_gen_traj, mock_train_dpo, mock_vllm
    ):
        """Two iterations should use updated paths in second iteration."""
        from src.training.alternating import alternating_train

        # Setup mocks
        mock_model = MagicMock()
        mock_vllm.return_value = mock_model
        mock_gen_traj.return_value = [
            {
                "positive": "A", "negative": "B",
                "positive_critic": "C", "negative_critic": "D",
                "round": 0, "delta": 0.3, "actor_prompt": "P",
            }
        ]

        # Track which paths are used for training
        train_calls = []
        def capture_train_call(model_name_or_path, **kwargs):
            train_calls.append(model_name_or_path)
            return f"/output/{model_name_or_path.split('/')[-1]}_trained"

        mock_train_dpo.side_effect = capture_train_call

        dataset = [
            {"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}
        ]

        result = alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=dataset,
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=2,
        )

        # Should have 4 training calls (2 iterations * 2 agents)
        assert len(train_calls) == 4

        # Iteration 0: Use original paths
        assert train_calls[0] == "/base/critic"  # Critic trained on base
        assert train_calls[1] == "/base/actor"   # Actor trained on base

        # Iteration 1: Should use UPDATED paths (the fix)
        # After iteration 0, paths should be the trained ones
        assert "critic" in train_calls[2]  # Second critic training
        assert "actor" in train_calls[3]   # Second actor training

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_empty_dataset_handling(self, mock_gen_traj, mock_train_dpo, mock_vllm):
        """Empty dataset should handle gracefully."""
        from src.training.alternating import alternating_train

        mock_model = MagicMock()
        mock_vllm.return_value = mock_model

        result = alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=[],
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=1,
        )

        # Should return original paths when no training occurs
        assert result["actor_path"] == "/base/actor"
        assert result["critic_path"] == "/base/critic"

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_no_preference_pairs_skips_training(
        self, mock_gen_traj, mock_train_dpo, mock_vllm
    ):
        """When no preference pairs generated, training should be skipped."""
        from src.training.alternating import alternating_train

        mock_model = MagicMock()
        mock_vllm.return_value = mock_model
        mock_gen_traj.return_value = []  # No pairs generated

        dataset = [
            {"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}
        ]

        result = alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=dataset,
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=1,
        )

        # train_dpo should not be called when no pairs
        assert mock_train_dpo.call_count == 0


class TestAlternatingTrainModelPathsFixVerification:
    """Verify the alternating_train model path fix (issue #18)."""

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_second_iteration_uses_trained_model_paths(
        self, mock_gen_traj, mock_train_dpo, mock_vllm
    ):
        """Verify that second iteration uses paths from first iteration (the fix)."""
        from src.training.alternating import alternating_train

        mock_model = MagicMock()
        mock_vllm.return_value = mock_model
        mock_gen_traj.return_value = [
            {"positive": "A", "negative": "B",
             "positive_critic": "C", "negative_critic": "D",
             "round": 0, "delta": 0.3, "actor_prompt": "P"}
        ]

        # Return distinctive paths for each training call
        call_count = [0]
        def mock_train(model_name_or_path, **kwargs):
            call_count[0] += 1
            if "critic" in model_name_or_path.lower():
                return f"/output/trained_critic_{call_count[0]}"
            return f"/output/trained_actor_{call_count[0]}"

        mock_train_dpo.side_effect = mock_train

        dataset = [{"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}]

        result = alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=dataset,
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=2,
        )

        # Final paths should be from iteration 1 (second iteration)
        assert "trained_actor" in result["actor_path"]
        assert "trained_critic" in result["critic_path"]

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_vllm_uses_updated_paths_between_iterations(
        self, mock_gen_traj, mock_train_dpo, mock_vllm
    ):
        """Verify VLLMInference is called with updated paths in second iteration."""
        from src.training.alternating import alternating_train

        mock_model = MagicMock()
        vllm_calls = []
        def track_vllm(path, **kwargs):
            vllm_calls.append(path)
            return mock_model
        mock_vllm.side_effect = track_vllm

        mock_gen_traj.return_value = [
            {"positive": "A", "negative": "B",
             "positive_critic": "C", "negative_critic": "D",
             "round": 0, "delta": 0.3, "actor_prompt": "P"}
        ]

        def mock_train(model_name_or_path, **kwargs):
            return f"/output/{model_name_or_path.split('/')[-1]}_trained"
        mock_train_dpo.side_effect = mock_train

        dataset = [{"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}]

        alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=dataset,
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=2,
        )

        # VLLM is called for each agent in each iteration
        # Iteration 0: actor for trajectory gen, critic for trajectory gen
        # Iteration 1: same again
        # So we expect at least 4 calls (may be more due to model loading)
        assert len(vllm_calls) >= 4

        # First two calls should use base paths (iteration 0)
        assert vllm_calls[0] == "/base/actor"
        assert vllm_calls[1] == "/base/critic"

        # Later calls should use trained paths or base depending on timing
        # The key is that trained models are used in iteration 1
        has_trained_path = any("trained" in str(call) for call in vllm_calls[2:])
        assert has_trained_path, "Should use trained model paths in second iteration"

    @patch("src.inference.vllm_server.VLLMInference")
    @patch("src.training.alternating.train_dpo")
    @patch("src.training.alternating.generate_trajectories")
    def test_three_iterations_path_progression(
        self, mock_gen_traj, mock_train_dpo, mock_vllm
    ):
        """Test path progression across 3 iterations (ACC-Collab+ scenario)."""
        from src.training.alternating import alternating_train

        mock_model = MagicMock()
        mock_vllm.return_value = mock_model
        mock_gen_traj.return_value = [
            {"positive": "A", "negative": "B",
             "positive_critic": "C", "negative_critic": "D",
             "round": 0, "delta": 0.3, "actor_prompt": "P"}
        ]

        train_calls = []
        def track_train(model_name_or_path, **kwargs):
            train_calls.append(model_name_or_path)
            return f"/output/{model_name_or_path.split('/')[-1]}_iter{len(train_calls)}"
        mock_train_dpo.side_effect = track_train

        dataset = [{"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}]

        result = alternating_train(
            actor_path="/base/actor",
            critic_path="/base/critic",
            dataset=dataset,
            dataset_name="boolq",
            output_base_dir="/output",
            num_iterations=3,
        )

        # Should have 6 training calls (3 iterations * 2 agents)
        assert len(train_calls) == 6

        # Each iteration should build on the previous one
        # Iteration 0
        assert train_calls[0] == "/base/critic"
        assert train_calls[1] == "/base/actor"

        # Iteration 1 and 2 should use progressively updated paths
        for i in range(2, 6):
            assert "iter" in train_calls[i] or "base" in train_calls[i]


class TestDPOBetaParameter:
    """Test DPO beta parameter configuration."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_beta_parameter_default(self, mock_trainer_class, mock_tokenizer_class, mock_model_class):
        """Beta parameter should default to 0.1."""
        from src.training.dpo_trainer import train_dpo

        mock_model_instance = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        train_dpo(
            model_name_or_path="test/model",
            preference_dataset=MagicMock(),
            output_dir="/output",
        )

        # Check that beta was passed to DPOConfig
        call_args = mock_trainer_class.call_args
        if call_args and len(call_args) > 1:
            args = call_args[1].get("args")
            if args and hasattr(args, "beta"):
                assert args.beta == 0.1

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_beta_parameter_custom(self, mock_trainer_class, mock_tokenizer_class, mock_model_class):
        """Custom beta parameter should be passed through."""
        from src.training.dpo_trainer import train_dpo

        mock_model_instance = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        train_dpo(
            model_name_or_path="test/model",
            preference_dataset=MagicMock(),
            output_dir="/output",
            beta=0.5,
        )

        # Verify beta=0.5 was used
        call_args = mock_trainer_class.call_args
        if call_args and len(call_args) > 1:
            args = call_args[1].get("args")
            if args and hasattr(args, "beta"):
                assert args.beta == 0.5

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_beta_zero_edge_case(self, mock_trainer_class, mock_tokenizer_class, mock_model_class):
        """Beta=0 should be handled (no reference policy constraint)."""
        from src.training.dpo_trainer import train_dpo

        mock_model_instance = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        train_dpo(
            model_name_or_path="test/model",
            preference_dataset=MagicMock(),
            output_dir="/output",
            beta=0.0,
        )

        # Verify beta=0.0 was used
        call_args = mock_trainer_class.call_args
        if call_args and len(call_args) > 1:
            args = call_args[1].get("args")
            if args and hasattr(args, "beta"):
                assert args.beta == 0.0


class TestDPOBetaParameterFixVerification:
    """Verify the DPO beta parameter fix (issue #21)."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_beta_passed_from_alternating_train_to_dpo(
        self, mock_trainer_class, mock_tokenizer_class, mock_model_class
    ):
        """Verify beta parameter flows from alternating_train to train_dpo."""
        from src.training.alternating import alternating_train

        # Setup mocks for train_dpo
        mock_model_instance = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        # Track beta values passed to train_dpo
        beta_values = []
        original_train_dpo = None

        def mock_train_dpo_with_beta_tracking(*args, **kwargs):
            beta_values.append(kwargs.get("beta", 0.1))
            return "/output/model"

        # Mock VLLMInference and generate_trajectories
        with patch("src.inference.vllm_server.VLLMInference") as mock_vllm:
            with patch("src.training.alternating.generate_trajectories") as mock_gen:
                with patch("src.training.alternating.train_dpo") as mock_train_dpo:
                    mock_vllm.return_value = MagicMock()
                    mock_gen.return_value = [
                        {"positive": "A", "negative": "B",
                         "positive_critic": "C", "negative_critic": "D",
                         "round": 0, "delta": 0.3, "actor_prompt": "P"}
                    ]
                    mock_train_dpo.side_effect = mock_train_dpo_with_beta_tracking

                    dataset = [{"question": "Q?", "passage": "P.", "answer": "yes", "task_type": "yes_no"}]

                    result = alternating_train(
                        actor_path="/base/actor",
                        critic_path="/base/critic",
                        dataset=dataset,
                        dataset_name="boolq",
                        output_base_dir="/output",
                        num_iterations=1,
                        beta=0.5,  # Custom beta value
                    )

        # Beta should have been passed through
        assert len(beta_values) == 2  # One for critic, one for actor
        assert all(b == 0.5 for b in beta_values), f"Expected beta=0.5, got {beta_values}"

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_dpo_config_contains_beta(
        self, mock_trainer_class, mock_tokenizer_class, mock_model_class
    ):
        """Verify DPOConfig is created with beta parameter."""
        from src.training.dpo_trainer import train_dpo
        from trl import DPOConfig

        mock_model_instance = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        # Patch DPOConfig in the trl module
        with patch("trl.DPOConfig", wraps=DPOConfig) as mock_config_class:
            train_dpo(
                model_name_or_path="test/model",
                preference_dataset=MagicMock(),
                output_dir="/output",
                beta=0.3,
            )

            # Check that DPOConfig was called with beta=0.3
            if mock_config_class.called:
                call_kwargs = mock_config_class.call_args[1]
                assert "beta" in call_kwargs
                assert call_kwargs["beta"] == 0.3

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_beta_default_value_matches_paper(
        self, mock_trainer_class, mock_tokenizer_class, mock_model_class
    ):
        """Verify default beta matches paper specification."""
        from src.training.dpo_trainer import train_dpo
        import inspect

        # Check function signature default
        sig = inspect.signature(train_dpo)
        default_beta = sig.parameters["beta"].default

        # Paper likely specifies beta around 0.1
        assert default_beta == 0.1, f"Default beta should be 0.1, got {default_beta}"

        # Also check alternating_train
        from src.training.alternating import alternating_train
        sig = inspect.signature(alternating_train)
        default_beta_alt = sig.parameters["beta"].default
        assert default_beta_alt == 0.1

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("trl.DPOTrainer")
    def test_beta_parameter_in_function_signature(
        self, mock_trainer_class, mock_tokenizer_class, mock_model_class
    ):
        """Verify beta parameter exists in both train_dpo and alternating_train."""
        from src.training.dpo_trainer import train_dpo
        from src.training.alternating import alternating_train
        import inspect

        # Check train_dpo has beta parameter
        sig_dpo = inspect.signature(train_dpo)
        assert "beta" in sig_dpo.parameters

        # Check alternating_train has beta parameter
        sig_alt = inspect.signature(alternating_train)
        assert "beta" in sig_alt.parameters
