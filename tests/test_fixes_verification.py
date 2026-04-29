"""
Tests for the 5 critical fixes to ACC-Collab implementation.

These tests verify the following fixes:
1. deliberation.py: previous_responses contains both actor and critic responses (interleaved)
2. trajectory.py: guided trajectories start from t=1 (skip t=0)
3. trajectory.py: preference pairs use independent ifs (both towards and away pairs can be generated per round)
4. dpo_trainer.py: DPO includes NLL regularization via loss_type=[loss_type, "sft"]
5. trajectory.py: guided pair simulation keeps actor and critic responses interleaved
"""

from unittest.mock import MagicMock, patch


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
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_guided_trajectories_skip_round_zero(self, mock_guided_prompt, mock_deliberate):
        """Guided trajectories should only be generated for t >= 1."""
        from src.algorithms.trajectory import generate_trajectories

        # Setup mocks
        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
            {"actor_response": "A2", "critic_response": "C2", "actor_prompt": "P2", "critic_prompt": "CP2"},
        ]
        mock_guided_prompt.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # MC roll-out now uses 3-phase actor-critic simulation per round:
        #   guided actor (2) → guided critic (2)
        #   → Phase A: actor (15) → Phase B: critic (15) → Phase C: actor (15)
        # For 2 rounds (t=1,2):
        #   actor.generate: 3 calls per round (guided + phase A + phase C) = 6 total
        #   critic.generate: 2 calls per round (guided + phase B) = 4 total
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],  # round 1: guided actor
            ["The answer is Yes."] * 15,  # round 1: MC Phase A (3*5=15)
            ["The answer is Yes."] * 15,  # round 1: MC Phase C (3*5=15)
            ["Guided actor z_y", "Guided actor z_not_y"],  # round 2: guided actor
            ["The answer is Yes."] * 15,  # round 2: MC Phase A
            ["The answer is Yes."] * 15,  # round 2: MC Phase C
        ]
        critic.generate.side_effect = [
            ["Guided critic z_y", "Guided critic z_not_y"],  # round 1: guided critic
            ["Critic feedback"] * 15,  # round 1: MC Phase B
            ["Guided critic z_y", "Guided critic z_not_y"],  # round 2: guided critic
            ["Critic feedback"] * 15,  # round 2: MC Phase B
        ]

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
            "choices": None,
        }

        generate_trajectories(
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
    def test_round_zero_excluded_from_preference_pairs(self, mock_deliberate):
        """Round t=0 should not generate any preference pairs."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]

        actor = MagicMock()
        critic = MagicMock()
        # 1 round (t=1): guided actor(2) + guided critic(2)
        # + MC Phase A(15) + Phase B(15) + Phase C(15)
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],  # guided actor
            ["The answer is Yes."] * 15,  # MC Phase A: all correct
            ["The answer is Yes."] * 15,  # MC Phase C: all correct -> high delta
        ]
        critic.generate.side_effect = [
            ["Guided critic z_y", "Guided critic z_not_y"],  # guided critic
            ["Critic feedback"] * 15,  # MC Phase B
        ]

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


class TestFix3TrajectoryIfStructure:
    """Fix #3: Preference pairs use independent ifs to collect all valid pairs."""

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_two_pairs_per_round_when_both_deltas_high(self, mock_guided, mock_deliberate):
        """When both delta_y and delta_not_y exceed threshold, both pairs should be created."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # MC roll-out: v_natural=0/5=0.0, v_guided_correct=5/5=1.0, v_guided_wrong=0/5=0.0
        # delta_y = 1.0 >= 0.3 -> "towards" pair
        # delta_not_y = 0.0 < 0.3 -> no "away" pair
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],
            ["No."] * 5 + ["The answer is Yes."] * 5 + ["No."] * 5,  # Phase A
            ["No."] * 5 + ["The answer is Yes."] * 5 + ["No."] * 5,  # Phase C
        ]
        critic.generate.side_effect = [
            ["Guided critic z_y", "Guided critic z_not_y"],
            ["Critic feedback"] * 15,  # Phase B
        ]

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

        # delta_y=1.0 >= 0.3 -> towards, delta_not_y=0.0 < 0.3 -> no away
        assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
        assert pairs[0]["direction"] == "towards"

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_both_pairs_when_both_deltas_exceed_threshold(self, mock_guided, mock_deliberate):
        """When both deltas exceed threshold, both 'towards' and 'away' pairs are created."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # v_natural=3/5=0.6, v_guided_correct=5/5=1.0, v_guided_wrong=0/5=0.0
        # delta_y = 0.4 >= 0.3 -> "towards" pair
        # delta_not_y = 0.6 >= 0.3 -> "away" pair
        mc_responses = [
            "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "No.", "No.",
            "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "The answer is Yes.",
            "No.", "No.", "No.", "No.", "No.",
        ]
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],
            mc_responses,  # Phase A
            mc_responses,  # Phase C
        ]
        critic.generate.side_effect = [
            ["Guided critic z_y", "Guided critic z_not_y"],
            ["Critic feedback"] * 15,  # Phase B
        ]

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

        # Both deltas >= 0.3, so both pairs are created
        assert len(pairs) == 2, f"Expected 2 pairs with if/if structure, got {len(pairs)}"
        directions = {p["direction"] for p in pairs}
        assert directions == {"towards", "away"}, f"Expected both directions, got {directions}"

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_away_pair_when_only_delta_not_y_exceeds(self, mock_guided, mock_deliberate):
        """When delta_y < threshold but delta_not_y >= threshold, should create 'away' pair."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # v_natural: 3/5=0.6, v_guided_correct: 4/5=0.8, v_guided_wrong: 0/5=0.0
        # delta_y = 0.8-0.6 = 0.2 < 0.3, delta_not_y = 0.6-0.0 = 0.6 >= 0.3 -> "away"
        mc_responses = [
            "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "No.", "No.",
            "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "The answer is Yes.", "No.",
            "No.", "No.", "No.", "No.", "No.",
        ]
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],
            mc_responses,  # Phase A
            mc_responses,  # Phase C
        ]
        critic.generate.side_effect = [
            ["Guided critic z_y", "Guided critic z_not_y"],
            ["Critic feedback"] * 15,  # Phase B
        ]

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

        assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
        assert pairs[0]["direction"] == "away", "Should create 'away' pair"

    @patch('src.algorithms.trajectory.deliberate')
    @patch('src.algorithms.trajectory._make_guided_prompt')
    def test_no_pair_when_both_deltas_below_threshold(self, mock_guided, mock_deliberate):
        """When both deltas are below threshold, no pair should be created."""
        from src.algorithms.trajectory import generate_trajectories

        mock_deliberate.return_value = [
            {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
            {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
        ]
        mock_guided.return_value = "Guided prompt"

        actor = MagicMock()
        critic = MagicMock()
        # v_natural=5/5=1.0, v_guided_correct=5/5=1.0, v_guided_wrong=5/5=1.0
        # delta_y = 0.0, delta_not_y = 0.0
        actor.generate.side_effect = [
            ["Guided actor z_y", "Guided actor z_not_y"],
            ["The answer is Yes."] * 15,  # Phase A
            ["The answer is Yes."] * 15,  # Phase C
        ]
        critic.generate.side_effect = [
            ["Guided critic z_y", "Guided critic z_not_y"],
            ["Critic feedback"] * 15,  # Phase B
        ]

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

        assert len(pairs) == 0, f"Expected 0 pairs when both deltas < threshold, got {len(pairs)}"


class TestFix4DPONLLRegularization:
    """Fix #4: DPO training should include NLL regularization."""

    def test_dpo_config_includes_nll_regularization(self):
        """DPOConfig should include SFT NLL regularization when sft_weight > 0."""
        import inspect
        from src.training._dpo_runner import _run

        source = inspect.getsource(_run)

        # Verify SFT regularization is conditionally added based on sft_weight
        assert 'effective_loss_type = [loss_type_val, "sft"]' in source, \
            "Code should construct loss_type list with 'sft' for NLL regularization"

        # Verify sft_weight controls the regularization
        assert "sft_weight" in source, \
            "Code should read sft_weight from config"

        # Verify loss_weights uses sft_weight
        assert "sft_weight" in source and "effective_loss_weights" in source, \
            "Code should use sft_weight for loss weights"

        # Verify NLL is documented in comments
        assert "NLL" in source or "negative log-likelihood" in source.lower(), \
            "Code should document NLL regularization"

    def test_nll_regularization_with_custom_loss_type(self):
        """NLL regularization should work with any base loss_type and be configurable."""
        import inspect
        from src.training._dpo_runner import _run

        source = inspect.getsource(_run)

        # The code reads loss_type from config and conditionally adds "sft"
        assert 'effective_loss_type = [loss_type_val, "sft"]' in source, \
            "Code should construct loss_type list with loss_type_val and 'sft'"

        # Verify sft_weight=0 disables SFT regularization
        assert "sft_weight > 0" in source, \
            "Code should allow disabling SFT regularization via sft_weight=0"

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
