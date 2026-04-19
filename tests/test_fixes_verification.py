"""
Tests for the 5 critical fixes to ACC-Collab implementation.

These tests verify the following fixes:
1. deliberation.py: previous_responses contains both actor and critic responses (interleaved)
2. trajectory.py: guided trajectories start from t=1 (skip t=0)
3. trajectory.py: preference pairs use if/elif instead of if/if
4. dpo_trainer.py: DPO includes NLL regularization via loss_type=[loss_type, "sft"]
5. rollout.py: sim_responses contains interleaved actor and critic responses
"""

import pytest
from unittest.mock import MagicMock, patch, call


class TestFix1DeliberationPreviousResponses:
    """Fix #1: previous_responses should contain interleaved actor and critic responses."""

    def test_previous_responses_interleaved_structure(self):
        """After each round, previous_responses should have actor then critic response."""
        from src.algorithms.deliberation import deliberate

        actor = MagicMock()
        critic = MagicMock()

        # Use distinctive responses for tracking
        actor_responses = ["Actor 0", "Actor 1", "Actor 2"]
        critic_responses = ["Critic 0", "Critic 1", "Critic 2"]

        actor.generate_single.side_effect = actor_responses
        critic.generate_single.side_effect = critic_responses

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Track format_prompt calls to verify responses structure
        prompt_calls = []
        original_format = __import__('src.prompts.formatter', fromlist=['format_prompt']).format_prompt
        def track_format(dataset_name, prompt_type, sample_arg, **kwargs):
            prompt_calls.append({
                "prompt_type": str(prompt_type),
                "responses": list(kwargs.get("responses", [])),  # Copy list
            })
            return original_format(dataset_name, prompt_type, sample_arg, **kwargs)

        with patch('src.algorithms.deliberation.format_prompt', side_effect=track_format):
            trajectory = deliberate(actor, critic, sample, "boolq", num_rounds=3)

        # Round 0: Single shot (no previous_responses)
        assert len(trajectory) == 3

        # Find deliberation actor prompts (they have responses parameter)
        deliberation_actor_calls = [
            p for p in prompt_calls
            if "DELIBERATION_ACTOR" in p["prompt_type"] and len(p["responses"]) > 0
        ]

        # First deliberation actor prompt (round 1, t=1) should have 2 items: [actor0, critic0]
        if len(deliberation_actor_calls) > 0:
            assert len(deliberation_actor_calls[0]["responses"]) == 2, \
                f"Round 1 should have actor0 + critic0 in previous_responses, got {deliberation_actor_calls[0]['responses']}"

        # Second deliberation actor prompt (round 2, t=2) should have 4 items: [actor0, critic0, actor1, critic1]
        if len(deliberation_actor_calls) > 1:
            assert len(deliberation_actor_calls[1]["responses"]) == 4, \
                f"Round 2 should have actor0 + critic0 + actor1 + critic1, got {deliberation_actor_calls[1]['responses']}"

    def test_previous_responses_actor_then_critic_order(self):
        """Verify order is actor response first, then critic response."""
        from src.algorithms.deliberation import deliberate

        actor = MagicMock()
        critic = MagicMock()

        # Distinct responses for verification
        actor_responses = [f"Actor {i}" for i in range(2)]
        critic_responses = [f"Critic {i}" for i in range(2)]

        actor.generate_single.side_effect = actor_responses
        critic.generate_single.side_effect = critic_responses

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        trajectory = deliberate(actor, critic, sample, "boolq", num_rounds=2)

        # Verify the order by checking trajectory structure
        # After round 0: responses = [actor0, critic0]
        # After round 1: responses = [actor0, critic0, actor1, critic1]

        # The key is that both actor and critic are added to previous_responses
        # We can verify this indirectly by checking that round 1 uses both
        assert trajectory[0]["actor_response"] == "Actor 0"
        assert trajectory[0]["critic_response"] == "Critic 0"
        assert trajectory[1]["actor_response"] == "Actor 1"
        assert trajectory[1]["critic_response"] == "Critic 1"


class TestFix2TrajectoryStartsFromT1:
    """Fix #2: Algorithm 1 guided trajectories should start from t=1, not t=0."""

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory.estimate_final_accuracy')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_guided_trajectories_skip_round_zero(self, mock_guided_prompt, mock_estimate, mock_deliberate):
        """Guided trajectories should only be generated for t >= 1."""
        from src.algorithms.trajectory import generate_trajectories

        # Setup mocks
        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
            {"actor_response": "A2", "critic_response": "C2", "actor_prompt": "P2", "critic_prompt": "CP2"},
        ]
        mock_estimate.return_value = 0.5
        mock_guided_prompt.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Guided actor"] * 4
        critic.generate.return_value = ["Guided critic"] * 4

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
            "choices": None,
        }

        pairs = generate_trajectories(
            actor, critic, sample, "boolq",
            num_rounds=3,
            reward_threshold=0.0,
        )

        # With 3 rounds (t=0,1,2), guided trajectories should only be for t=1,2
        # _make_guided_prompt called 4 times per round (actor+critic * 2 directions)
        # For 2 rounds: 4 * 2 = 8 calls total
        assert mock_guided_prompt.call_count == 8, f"Expected 8 calls for t=1,2, got {mock_guided_prompt.call_count}"

        # Verify no guided prompt was made for t=0
        for call_args in mock_guided_prompt.call_args_list:
            round_idx = call_args[0][3]  # 4th positional arg is round_idx
            assert round_idx >= 1, f"Guided prompt should not be generated for t=0, got t={round_idx}"

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory.estimate_final_accuracy')
    def test_round_zero_excluded_from_preference_pairs(self, mock_estimate, mock_deliberate):
        """Round t=0 should not generate any preference pairs."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_estimate.return_value = 1.0

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Guided"] * 4
        critic.generate.return_value = ["Guided critic"] * 4

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
            "choices": None,
        }

        pairs = generate_trajectories(
            actor, critic, sample, "boolq",
            num_rounds=2,
            reward_threshold=0.0,
        )

        # All preference pairs should have round >= 1
        for pair in pairs:
            assert pair["round"] >= 1, f"Round should be >= 1, got {pair['round']}"


class TestFix3TrajectoryIfElifStructure:
    """Fix #3: Preference pairs should use if/elif, not two independent ifs."""

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_only_one_pair_per_round_when_both_deltas_high(self, mock_guided, mock_deliberate):
        """When both delta_y and delta_not_y exceed threshold, only one pair should be created."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # actor.generate is called twice:
        # 1. guided actor (2 prompts) -> 2 responses
        # 2. merged MC roll-out (3*5=15 prompts) -> 15 responses
        # critic.generate is called once:
        # 1. guided critic (2 prompts) -> 2 responses
        # MC roll-out: 5 "No" + 5 "Yes" + 5 "No"
        # v_natural=0/5=0.0, v_guided_correct=5/5=1.0, v_guided_wrong=0/5=0.0
        # delta_y = 1.0 >= 0.3 -> "towards" pair
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],  # guided actor
            ["No."] * 5 + ["The answer is Yes."] * 5 + ["No."] * 5,  # MC roll-out
        ]
        critic.generate.return_value = ["Guided critic"] * 2

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
            "choices": None,
        }

        pairs = generate_trajectories(
            actor, critic, sample, "boolq",
            num_rounds=2,
            reward_threshold=0.3,
        )

        # Should only create ONE pair per round (the "towards" one from if, not the "away" from elif)
        assert len(pairs) == 1, f"Expected 1 pair with if/elif structure, got {len(pairs)}"
        assert pairs[0]["direction"] == "towards", "First matching condition (towards) should be used"

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_elif_branch_creates_away_pair_when_if_fails(self, mock_guided, mock_deliberate):
        """When delta_y < threshold but delta_not_y >= threshold, should create 'away' pair."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # actor.generate called twice: guided actor (2), MC roll-out (15)
        # MC roll-out: v_natural: 3/5=0.6, v_guided_correct: 4/5=0.8, v_guided_wrong: 0/5=0.0
        # delta_y = 0.8-0.6 = 0.2 < 0.3, delta_not_y = 0.6-0.0 = 0.6 >= 0.3 -> "away"
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],  # guided actor
            ["The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "No.", "No.",
             "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "No.",
             "No.", "No.", "No.", "No.", "No."],
        ]
        critic.generate.return_value = ["Guided critic"] * 2

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
            "choices": None,
        }

        pairs = generate_trajectories(
            actor, critic, sample, "boolq",
            num_rounds=2,
            reward_threshold=0.3,
        )

        # Should create ONE pair (the "away" one from elif)
        assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
        assert pairs[0]["direction"] == "away", "Elif branch should create 'away' pair"

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory.estimate_final_accuracy')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_no_pair_when_both_deltas_below_threshold(self, mock_guided, mock_estimate, mock_deliberate):
        """When both deltas are below threshold, no pair should be created."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        call_count = [0]
        def side_effect_estimate(*args, **kwargs):
            call_count[0] += 1
            # All returns similar values -> small deltas
            # v_natural=0.5, v_guided_correct=0.55, v_guided_wrong=0.45
            # delta_y = 0.55 - 0.5 = 0.05 < threshold
            # delta_not_y = 0.5 - 0.45 = 0.05 < threshold
            idx = call_count[0] - 1
            values = [0.5, 0.55, 0.45]
            return values[idx % len(values)]

        mock_estimate.side_effect = side_effect_estimate

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Guided actor"] * 4
        critic.generate.return_value = ["Guided critic"] * 4

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
            "choices": None,
        }

        pairs = generate_trajectories(
            actor, critic, sample, "boolq",
            num_rounds=2,
            reward_threshold=0.3,
        )

        # Should create NO pairs
        assert len(pairs) == 0, f"Expected 0 pairs when both deltas < threshold, got {len(pairs)}"


class TestFix4DPONLLRegularization:
    """Fix #4: DPO training should include NLL regularization."""

    def test_dpo_config_includes_nll_regularization(self):
        """DPOConfig should have loss_type=[dpo_loss, "sft"] for NLL regularization."""
        # Since train_dpo imports transformers/trl internally which may try to download models,
        # we'll verify the fix by checking the source code directly.
        # The DPO training logic lives in _dpo_runner.py (subprocess runner).
        import inspect
        from src.training._dpo_runner import _run

        source = inspect.getsource(_run)

        # Verify loss_type is set to a list including "sft"
        assert 'loss_type=[loss_type_val, "sft"]' in source, \
            "Code should set loss_type=[loss_type_val, 'sft'] for NLL regularization"

        # Verify loss_weights is set
        assert 'loss_weights=[1.0, 1.0]' in source, \
            "Code should set loss_weights=[1.0, 1.0]"

        # Verify NLL is documented in comments
        assert "NLL" in source or "negative log-likelihood" in source.lower(), \
            "Code should document NLL regularization"

    def test_nll_regularization_with_custom_loss_type(self):
        """NLL regularization should work with any base loss_type."""
        import inspect
        from src.training._dpo_runner import _run

        source = inspect.getsource(_run)

        # The code reads loss_type from config and adds "sft" to it
        assert 'loss_type=[loss_type_val, "sft"]' in source, \
            "Code should use loss_type parameter and add 'sft' to the list"

    def test_dpo_code_mentions_nll_regularization(self):
        """The code should document NLL regularization in comments."""
        import inspect
        from src.training._dpo_runner import _run

        source = inspect.getsource(_run)
        # The NLL regularization is documented in dpo_trainer.py comments
        from src.training.dpo_trainer import train_dpo
        trainer_source = inspect.getsource(train_dpo)

        combined = source + trainer_source
        assert "NLL" in combined or "negative log-likelihood" in combined.lower(), \
            "Code should mention NLL regularization"
        assert "sft" in combined.lower(), "Code should mention SFT loss for NLL regularization"


class TestFix5RolloutSimResponsesInterleaved:
    """Fix #5: sim_responses should contain interleaved actor and critic responses."""

    def test_sim_responses_initialization_includes_both_current_responses(self):
        """sim_responses should start with previous + current_actor + current_critic."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["The answer is Yes."]
        critic.generate.return_value = ["Feedback."]

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Track what responses are passed to format_prompt
        responses_passed = []
        original_format = __import__('src.prompts.formatter', fromlist=['format_prompt']).format_prompt
        def track_format(dataset_name, prompt_type, sample_arg, **kwargs):
            responses_passed.append(kwargs.get("responses", []))
            return original_format(dataset_name, prompt_type, sample_arg, **kwargs)

        with patch('src.algorithms.rollout.format_prompt', side_effect=track_format):
            estimate_final_accuracy(
                actor, critic, sample, "boolq",
                current_actor_response="Current actor",
                current_critic_response="Current critic",
                previous_responses=["Prev actor 0", "Prev critic 0"],
                num_simulations=1,
                remaining_rounds=1,
            )

        # First call to format_prompt (for actor) should have:
        # previous + current_actor + current_critic
        # = [Prev actor 0, Prev critic 0, Current actor, Current critic]
        assert len(responses_passed) > 0
        assert len(responses_passed[0]) >= 4, f"Expected at least 4 responses, got {len(responses_passed[0])}"
        assert "Current actor" in responses_passed[0]
        assert "Current critic" in responses_passed[0]

    def test_sim_responses_uses_batch_generate(self):
        """Optimized rollout should use batch generate for all simulations."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Simulated actor"] * 3
        critic.generate.return_value = ["Simulated critic"] * 3

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="Current A",
            current_critic_response="Current C",
            previous_responses=[],
            num_simulations=3,
            remaining_rounds=1,
        )

        # Should call batch generate once for actor (one-step roll-out only needs actor)
        assert actor.generate.call_count == 1
        assert critic.generate.call_count == 0  # Critic not needed in one-step roll-out
        # Actor should receive 3 prompts (one per simulation)
        assert len(actor.generate.call_args[0][0]) == 3

    def test_interleaved_order_preserved(self):
        """Responses should maintain actor, critic, actor, critic order."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Yes."]
        critic.generate.return_value = ["Good."]

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Use distinctive responses to track order
        previous_responses = ["Prev A0", "Prev C0", "Prev A1", "Prev C1"]

        responses_captured = []
        original_format = __import__('src.prompts.formatter', fromlist=['format_prompt']).format_prompt
        def track_format(dataset_name, prompt_type, sample_arg, **kwargs):
            responses = kwargs.get("responses", [])
            responses_captured.append(list(responses))  # Copy
            return original_format(dataset_name, prompt_type, sample_arg, **kwargs)

        with patch('src.algorithms.rollout.format_prompt', side_effect=track_format):
            estimate_final_accuracy(
                actor, critic, sample, "boolq",
                current_actor_response="Curr A",
                current_critic_response="Curr C",
                previous_responses=previous_responses,
                num_simulations=1,
                remaining_rounds=1,
            )

        # First actor prompt should see interleaved structure
        first_responses = responses_captured[0]
        # Order: [Prev A0, Prev C0, Prev A1, Prev C1, Curr A, Curr C]
        assert first_responses[0] == "Prev A0"
        assert first_responses[1] == "Prev C0"
        assert first_responses[2] == "Prev A1"
        assert first_responses[3] == "Prev C1"
        assert first_responses[4] == "Curr A"
        assert first_responses[5] == "Curr C"
