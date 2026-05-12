"""Tests for the natural multi-agent society stack."""

from __future__ import annotations

from src.society.agent_registry import (
    AgentConfig,
    AgentRegistry,
    AgentRole,
    CriticSkill,
    ReasoningStyle,
    resolve_critic_skill,
    resolve_reasoning_style,
)
from src.society.multi_deliberation import (
    DeliberationRound,
    _critic_aware_consensus,
    multi_agent_deliberate_single_gpu,
)
from src.society.router import CriticFeedback, CriticRouter, build_critic_feedback


class FakeEngine:
    supports_lora = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        count = len(prompts)
        out = self.responses[:count]
        self.responses = self.responses[count:]
        assert len(out) == count
        return out


def test_agent_registry_builds_default_society():
    registry = AgentRegistry.create_default(base_model_path="base")

    assert [actor.reasoning_style for actor in registry.list_actors()] == [
        ReasoningStyle.DIRECT,
        ReasoningStyle.EVIDENCE,
        ReasoningStyle.ELIMINATION,
    ]
    assert [critic.error_specialty for critic in registry.list_critics()] == [
        CriticSkill.COMPUTATION,
        CriticSkill.REASONING,
        CriticSkill.KNOWLEDGE,
        CriticSkill.GROUNDING,
        CriticSkill.VERIFICATION,
    ]


def test_enum_resolution_is_strict_but_case_insensitive():
    assert resolve_reasoning_style("Evidence") is ReasoningStyle.EVIDENCE
    assert resolve_critic_skill("factual") is CriticSkill.KNOWLEDGE


def test_build_critic_feedback_parses_field_level_judgement():
    critic = AgentConfig(
        name="critic_reasoning",
        role=AgentRole.CRITIC,
        model_path="base",
        error_specialty=CriticSkill.REASONING,
    )

    feedback = build_critic_feedback(
        critic,
        """The actor overlooks the decisive option wording.

Judgement:
Answer correct: no
Suggested answer: C
Confidence: 0.82""",
        task_type="multiple_choice",
    )

    assert feedback.critic_name == "critic_reasoning"
    assert feedback.skill == "reasoning"
    assert feedback.answer_correct == "no"
    assert feedback.suggested_answer == "C"
    assert feedback.confidence == 0.82
    assert feedback.usable_for_feedback
    assert feedback.usable_for_routing
    assert feedback.usable_for_consensus


def test_router_selects_natural_feedback_without_schema_filter():
    low_no_conf = CriticFeedback(
        critic_name="critic_a",
        skill="reasoning",
        critique="Useful but low-information feedback.",
        confidence=None,
        answer_correct="unknown",
        suggested_answer=None,
        usable_for_feedback=True,
        usable_for_routing=False,
        usable_for_consensus=False,
    )
    high = CriticFeedback(
        critic_name="critic_b",
        skill="grounding",
        critique="The passage contradicts the actor's answer.",
        confidence=0.9,
        answer_correct="no",
        suggested_answer="B",
        usable_for_feedback=True,
        usable_for_routing=True,
        usable_for_consensus=True,
    )

    decision = CriticRouter(top_k=1).route([low_no_conf, high])

    assert decision.selected_feedbacks == [high]
    assert "The passage contradicts" in decision.feedback_text
    assert "confidence=" not in decision.feedback_text
    assert "schema_valid" not in decision.feedback_text


def test_router_default_score_parameter_affects_missing_confidence_feedback():
    no_conf = CriticFeedback(
        critic_name="critic_no_conf",
        skill="reasoning",
        critique="No confidence field, but useful critique.",
        confidence=None,
        answer_correct="unknown",
        suggested_answer=None,
        usable_for_feedback=True,
        usable_for_routing=False,
        usable_for_consensus=False,
    )
    low_conf = CriticFeedback(
        critic_name="critic_low_conf",
        skill="verification",
        critique="Low confidence critique.",
        confidence=0.2,
        answer_correct="unknown",
        suggested_answer=None,
        usable_for_feedback=True,
        usable_for_routing=True,
        usable_for_consensus=False,
    )

    decision = CriticRouter(top_k=1, default_score=0.8).route([no_conf, low_conf])

    assert decision.selected_feedbacks == [no_conf]


def test_critic_aware_consensus_uses_actor_votes_and_critic_signals():
    feedback = CriticFeedback(
        critic_name="critic_reasoning",
        skill="reasoning",
        critique="Actor should switch to C.",
        confidence=0.9,
        answer_correct="no",
        suggested_answer="C",
        usable_for_feedback=True,
        usable_for_routing=True,
        usable_for_consensus=True,
    )
    round_data = DeliberationRound(
        round_num=0,
        actor_responses={"actor_direct": "The final result is A."},
        actor_answers={"actor_direct": "A"},
        actor_answer_sources={"actor_direct": "final_result"},
        actor_parse_confidence={"actor_direct": 1.0},
        critic_raw_responses={"actor_direct": {"critic_reasoning": feedback.raw_response}},
        critic_feedbacks={"actor_direct": {"critic_reasoning": feedback}},
        routed_feedbacks={"actor_direct": [feedback]},
    )

    answer, confidence, weights = _critic_aware_consensus(round_data, "multiple_choice")

    assert answer == "A"
    assert weights["A"] == 1.0
    assert weights["C"] == 0.9
    assert confidence > 0.5


def test_critic_aware_consensus_can_use_selected_critics_only():
    selected = CriticFeedback(
        critic_name="critic_selected",
        skill="verification",
        critique="Actor answer is acceptable.",
        confidence=0.2,
        answer_correct="yes",
        suggested_answer="A",
        usable_for_feedback=True,
        usable_for_routing=True,
        usable_for_consensus=True,
    )
    unselected_1 = CriticFeedback(
        critic_name="critic_unselected_1",
        skill="reasoning",
        critique="Actor should switch to C.",
        confidence=0.9,
        answer_correct="no",
        suggested_answer="C",
        usable_for_feedback=True,
        usable_for_routing=True,
        usable_for_consensus=True,
    )
    unselected_2 = CriticFeedback(
        critic_name="critic_unselected_2",
        skill="grounding",
        critique="C is better supported.",
        confidence=0.8,
        answer_correct="no",
        suggested_answer="C",
        usable_for_feedback=True,
        usable_for_routing=True,
        usable_for_consensus=True,
    )
    round_data = DeliberationRound(
        round_num=0,
        actor_responses={"actor_direct": "The final result is A."},
        actor_answers={"actor_direct": "A"},
        actor_answer_sources={"actor_direct": "final_result"},
        actor_parse_confidence={"actor_direct": 1.0},
        critic_raw_responses={
            "actor_direct": {
                "critic_selected": selected.raw_response,
                "critic_unselected_1": unselected_1.raw_response,
                "critic_unselected_2": unselected_2.raw_response,
            }
        },
        critic_feedbacks={
            "actor_direct": {
                "critic_selected": selected,
                "critic_unselected_1": unselected_1,
                "critic_unselected_2": unselected_2,
            }
        },
        routed_feedbacks={"actor_direct": [selected]},
    )

    all_answer, _, all_weights = _critic_aware_consensus(round_data, "multiple_choice")
    selected_answer, _, selected_weights = _critic_aware_consensus(
        round_data,
        "multiple_choice",
        consensus_uses_selected_critics_only=True,
    )

    assert all_answer == "C"
    assert abs(all_weights["C"] - 1.7) < 1e-12
    assert selected_answer == "A"
    assert selected_weights == {"A": 1.2}


def test_multi_agent_deliberation_keeps_raw_actor_response_and_routes_feedback():
    actors = [
        AgentConfig(
            name="actor_direct",
            role=AgentRole.ACTOR,
            model_path="base",
            reasoning_style=ReasoningStyle.DIRECT,
        )
    ]
    critics = [
        AgentConfig(
            name="critic_verification",
            role=AgentRole.CRITIC,
            model_path="base",
            error_specialty=CriticSkill.VERIFICATION,
        )
    ]
    engine = FakeEngine([
        "Reasoning text.\nThe final result is A.",
        """The answer is consistent.

Judgement:
Answer correct: yes
Suggested answer: A
Confidence: 0.7""",
        "Revised reasoning.\nThe final result is B.",
        """The actor changed incorrectly.

Judgement:
Answer correct: no
Suggested answer: A
Confidence: 0.9""",
    ])

    result = multi_agent_deliberate_single_gpu(
        inference_engine=engine,
        actors=actors,
        critics=critics,
        sample={
            "question": "Pick A.",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "task_type": "multiple_choice",
        },
        dataset_name="mmlu",
        num_rounds=2,
        max_tokens=16,
        temperature=0.0,
    )

    assert result.rounds[0].actor_responses["actor_direct"] == "Reasoning text.\nThe final result is A."
    assert result.rounds[0].actor_answers["actor_direct"] == "A"
    assert result.rounds[0].actor_answer_sources["actor_direct"] == "final_result"
    assert result.rounds[0].routed_feedbacks["actor_direct"][0].critique.startswith("The answer is consistent")
    assert result.rounds[1].actor_answers["actor_direct"] == "B"
    assert result.rounds[1].consensus_weights["A"] == 0.9
    assert result.rounds[1].consensus_weights["B"] == 1.0
    assert result.consensus_answer == "B"


def test_route_feedback_to_actor_can_be_disabled_without_disabling_routing():
    actors = [
        AgentConfig(
            name="actor_direct",
            role=AgentRole.ACTOR,
            model_path="base",
            reasoning_style=ReasoningStyle.DIRECT,
        )
    ]
    critics = [
        AgentConfig(
            name="critic_verification",
            role=AgentRole.CRITIC,
            model_path="base",
            error_specialty=CriticSkill.VERIFICATION,
        )
    ]
    engine = FakeEngine([
        "Initial reasoning.\nThe final result is A.",
        """The option mapping is wrong; compare B and C.

Judgement:
Answer correct: no
Suggested answer: C
Confidence: 0.9""",
        "Revision without feedback.\nThe final result is A.",
        """The actor stayed with A.

Judgement:
Answer correct: no
Suggested answer: C
Confidence: 0.8""",
    ])

    result = multi_agent_deliberate_single_gpu(
        inference_engine=engine,
        actors=actors,
        critics=critics,
        sample={
            "question": "Pick C.",
            "choices": ["A", "B", "C", "D"],
            "answer": "C",
            "task_type": "multiple_choice",
        },
        dataset_name="mmlu",
        num_rounds=2,
        max_tokens=16,
        temperature=0.0,
        route_feedback_to_actor=False,
    )

    assert result.rounds[0].routed_feedbacks["actor_direct"][0].critique.startswith(
        "The option mapping is wrong"
    )
    assert result.rounds[0].routed_feedback_texts["actor_direct"] == ""
    assert "The option mapping is wrong" not in engine.prompts[2]
    assert "No critic feedback was selected." in engine.prompts[2]
