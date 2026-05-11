from src.parsing.answer_extractor import extract_answer
from src.parsing.critic_parser import parse_critic_response
from src.prompts.prompt_builder import build_simple_actor_prompt, build_simple_critic_prompt
from src.society.agent_registry import AgentConfig, AgentRole, CriticSkill, ReasoningStyle
from src.society.router import CriticRouter, build_critic_feedback


def test_actor_prompt_uses_final_result_anchor():
    prompt = build_simple_actor_prompt(
        {
            "question": "Which option is correct?",
            "choices": ["one", "two", "three", "four"],
            "task_type": "multiple_choice",
        },
        "mmlu",
        style=ReasoningStyle.DIRECT,
    )
    assert "The final result is <answer>." in prompt
    assert "Question:" in prompt
    assert "Options:" in prompt
    assert "FINAL_ANSWER" not in prompt


def test_answer_extractor_prioritizes_final_result_and_tail_only():
    response = (
        "Option A is tempting and option B is mentioned in the reasoning.\n"
        "The final result is C."
    )
    extracted = extract_answer(response, "multiple_choice")
    assert extracted.answer == "C"
    assert extracted.source == "final_result"
    assert extracted.confidence == 1.0


def test_critic_parser_reads_natural_judgement_block():
    response = """The actor misses the wording in the option and should switch.

Judgement:
Answer correct: no
Suggested answer: C
Confidence: 0.82"""
    parsed = parse_critic_response(response, "multiple_choice")
    assert parsed.critique.startswith("The actor misses")
    assert parsed.answer_correct == "no"
    assert parsed.suggested_answer == "C"
    assert parsed.confidence == 0.82
    assert parsed.usable_for_feedback
    assert parsed.usable_for_consensus


def test_critic_parser_treats_unknown_suggestion_as_non_consensus_signal():
    response = """The actor may be right, but the issue is ambiguous.

Judgement:
Answer correct: uncertain
Suggested answer: unknown
Confidence: 0.4"""

    parsed = parse_critic_response(response, "multiple_choice")

    assert parsed.suggested_answer == "unknown"
    assert parsed.has_suggested_answer is False
    assert parsed.usable_for_feedback
    assert parsed.usable_for_consensus is False


def test_router_routes_natural_feedback_without_schema_filter():
    critic = AgentConfig(
        name="critic_reasoning",
        role=AgentRole.CRITIC,
        model_path="base",
        error_specialty=CriticSkill.REASONING,
    )
    feedback = build_critic_feedback(
        critic,
        """This is useful natural critique.

Judgement:
Answer correct: no
Suggested answer: B
Confidence: 0.7""",
        task_type="multiple_choice",
    )
    routed = CriticRouter(top_k=1).route([feedback])
    assert routed.selected_feedbacks == [feedback]
    assert "weight=" not in routed.feedback_text
    assert "This is useful natural critique" in routed.feedback_text


def test_critic_prompt_uses_natural_judgement_not_schema():
    prompt = build_simple_critic_prompt(
        {
            "question": "Is the statement supported?",
            "passage": "The passage says yes.",
            "task_type": "yes_no",
        },
        "boolq",
        "The final result is Yes.",
        skill=CriticSkill.GROUNDING,
    )
    assert "Judgement:" in prompt
    assert "Answer correct: yes/no/uncertain" in prompt
    assert "[Answer_Correct" not in prompt
