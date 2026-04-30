"""Tests for trajectory generation and preference data."""

from unittest.mock import MagicMock, patch

from src.algorithms.trajectory import generate_trajectories_batch
from src.trajectory.preference import build_preference_dataset


class TestBatchedTrajectoryGeneration:
    """Test batched Algorithm 1 trajectory generation."""

    @patch("src.algorithms.trajectory.format_prompt", return_value="rollout prompt")
    @patch("src.algorithms.trajectory.deliberate_batch")
    @patch("src.algorithms.trajectory._make_guided_prompt")
    def test_guided_and_mc_rollout_are_cross_sample_batched(
        self,
        mock_guided,
        mock_deliberate_batch,
        mock_format_prompt,
    ):
        mock_guided.side_effect = lambda *args, **kwargs: f"{kwargs.get('agent', 'actor')} prompt"
        samples = [
            {"question": "Q1?", "answer": "yes", "task_type": "yes_no"},
            {"question": "Q2?", "answer": "no", "task_type": "yes_no"},
        ]
        mock_deliberate_batch.return_value = [
            [
                {"actor_response": "A0", "critic_response": "C0", "actor_prompt": "P0", "critic_prompt": "CP0"},
                {"actor_response": "A1", "critic_response": "C1", "actor_prompt": "P1", "critic_prompt": "CP1"},
            ],
            [
                {"actor_response": "B0", "critic_response": "D0", "actor_prompt": "P2", "critic_prompt": "CP2"},
                {"actor_response": "B1", "critic_response": "D1", "actor_prompt": "P3", "critic_prompt": "CP3"},
            ],
        ]

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.side_effect = [
            ["GY1", "GW1", "GY2", "GW2"],
            ["No."] * 12,
            ["No."] * 4 + ["The answer is Yes."] * 4 + ["No."] * 4,
        ]
        critic.generate.side_effect = [
            ["CY1", "CW1", "CY2", "CW2"],
            ["Feedback"] * 12,
        ]

        pairs = generate_trajectories_batch(
            actor,
            critic,
            samples,
            "boolq",
            num_rounds=2,
            num_simulations=2,
            reward_threshold=0.0,
            batch_size=2,
        )

        assert len(pairs) == 2
        assert len(actor.generate.call_args_list[0].args[0]) == 4
        assert len(critic.generate.call_args_list[0].args[0]) == 4
        assert len(actor.generate.call_args_list[1].args[0]) == 12
        assert len(critic.generate.call_args_list[1].args[0]) == 12
        assert len(actor.generate.call_args_list[2].args[0]) == 12


class TestPreferenceBuilder:
    """Test preference pair construction."""

    def test_build_actor_preferences(self):
        pairs = [
            {
                "actor_prompt": "Prompt 1",
                "positive": "Good answer",
                "negative": "Bad answer",
                "positive_critic": "Good feedback",
                "negative_critic": "Bad feedback",
                "round": 0,
                "delta": 0.3,
                "direction": "towards",
            },
            {
                "actor_prompt": "Prompt 2",
                "positive": "Better answer",
                "negative": "Worse answer",
                "positive_critic": "Better feedback",
                "negative_critic": "Worse feedback",
                "round": 1,
                "delta": 0.1,
                "direction": "away",
            },
        ]
        dataset = build_preference_dataset(pairs, agent="actor")
        assert len(dataset) == 2
        assert dataset[0]["chosen"] == "Good answer"
        assert dataset[0]["rejected"] == "Bad answer"

    def test_build_critic_preferences(self):
        pairs = [
            {
                "critic_prompt": "Critic prompt",
                "positive": "A",
                "negative": "B",
                "positive_critic": "Good feedback",
                "negative_critic": "Bad feedback",
                "round": 0,
                "delta": 0.2,
            },
        ]
        dataset = build_preference_dataset(pairs, agent="critic")
        assert len(dataset) == 1
        assert dataset[0]["chosen"] == "Good feedback"
        assert dataset[0]["rejected"] == "Bad feedback"

    def test_min_delta_filter(self):
        pairs = [
            {"positive": "A", "negative": "B",
             "positive_critic": "C", "negative_critic": "D",
             "round": 0, "delta": 0.3, "actor_prompt": "P"},
            {"positive": "E", "negative": "F",
             "positive_critic": "G", "negative_critic": "H",
             "round": 0, "delta": 0.05, "actor_prompt": "Q"},
        ]
        dataset = build_preference_dataset(pairs, min_delta=0.1, agent="actor")
        assert len(dataset) == 1
        assert dataset[0]["chosen"] == "A"

    def test_empty_pairs(self):
        dataset = build_preference_dataset([], agent="actor")
        assert len(dataset) == 0
