"""Tests for trajectory generation and preference data."""

import pytest
from unittest.mock import MagicMock

from src.trajectory.preference import build_preference_dataset


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
