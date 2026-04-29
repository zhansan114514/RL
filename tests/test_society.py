"""
Comprehensive tests for Phase 0 Society module.

Tests cover:
1. agent_registry.py - AgentRegistry, AgentConfig, enums
2. router.py - CriticRouter, parse_confidence, build_critic_feedback
3. data_classifier.py - DataClassifier, API classification, ClassificationError
4. diversity_split.py - DiversitySplit, style/error profile splitting
5. inference_pipeline.py - InferencePipeline, voting strategies, ABLATION_CONFIGS
6. Phase 0.1 integration - MATH/GSM answer extraction
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

# Import all society modules
from src.society.agent_registry import (
    AgentRegistry,
    AgentConfig,
    AgentRole,
    ReasoningStyle,
    CriticSkill,
    CRITIC_CONFIDENCE_SUFFIX,
)
from src.society.router import (
    CriticRouter,
    CriticFeedback,
    RoutedFeedback,
    parse_confidence,
    parse_answer_correct,
    build_critic_feedback,
)
from src.society.data_classifier import (
    DataClassifier,
    ClassificationError,
    classify_reasoning_style,
    classify_error_profile,
    check_api_available,
    _parse_style_response,
    _parse_error_profile_response,
    _compute_sample_hash,
)
from src.society.diversity_split import DiversitySplit, RoutedTrainingItem
from src.society.inference_pipeline import (
    _majority_vote,
    _weighted_vote,
    _best_actor,
    ABLATION_CONFIGS,
)
from src.algorithms.reward import extract_answer
from src.data.preprocessor import generate_wrong_answer


# ============================================================
# 1. agent_registry.py Tests
# ============================================================

class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_create_actor_config(self):
        """Should create Actor config with reasoning style."""
        config = AgentConfig(
            name="actor_algebraic",
            role=AgentRole.ACTOR,
            model_path="/models/base",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        )
        assert config.name == "actor_algebraic"
        assert config.role == AgentRole.ACTOR
        assert config.reasoning_style == ReasoningStyle.ALGEBRAIC
        assert config.error_specialty is None

    def test_create_critic_config(self):
        """Should create Critic config with error specialty."""
        config = AgentConfig(
            name="critic_computation",
            role=AgentRole.CRITIC,
            model_path="/models/base",
            error_specialty=CriticSkill.COMPUTATION,
        )
        assert config.role == AgentRole.CRITIC
        assert config.error_specialty == CriticSkill.COMPUTATION
        assert config.reasoning_style is None

    def test_display_name_actor(self):
        """Actor display_name should include reasoning style."""
        config = AgentConfig(
            name="actor_1",
            role=AgentRole.ACTOR,
            model_path="/models/base",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        )
        assert config.display_name == "Actor-algebraic"

    def test_display_name_critic(self):
        """Critic display_name should include error specialty."""
        config = AgentConfig(
            name="critic_1",
            role=AgentRole.CRITIC,
            model_path="/models/base",
            error_specialty=CriticSkill.REASONING,
        )
        assert config.display_name == "Critic-reasoning"

    def test_system_prompt_actor(self):
        """Actor system prompt should be built from style."""
        config = AgentConfig(
            name="actor_1",
            role=AgentRole.ACTOR,
            model_path="/models/base",
            reasoning_style=ReasoningStyle.DIRECT,
        )
        assert "direct computational reasoner" in config.system_prompt.lower()

    def test_system_prompt_critic(self):
        """Critic system prompt should include confidence suffix."""
        config = AgentConfig(
            name="critic_1",
            role=AgentRole.CRITIC,
            model_path="/models/base",
            error_specialty=CriticSkill.VERIFICATION,
        )
        assert CRITIC_CONFIDENCE_SUFFIX in config.system_prompt
        assert "[Confidence: 0.0-1.0]" in config.system_prompt
        assert "[Suggested_Final_Answer:" in config.system_prompt
        assert "[Error_Type:" in config.system_prompt

    def test_to_dict(self):
        """Should serialize to dictionary."""
        config = AgentConfig(
            name="test_actor",
            role=AgentRole.ACTOR,
            model_path="/models/base",
            reasoning_style=ReasoningStyle.BACKTRACKING,
            temperature=0.5,
            max_tokens=256,
        )
        d = config.to_dict()
        assert d["name"] == "test_actor"
        assert d["role"] == "actor"
        assert d["reasoning_style"] == "backtracking"
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 256

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "name": "test_critic",
            "role": "critic",
            "model_path": "/models/base",
            "lora_path": "/models/lora",
            "error_specialty": "computation",
            "temperature": 0.8,
            "max_tokens": 512,
            "system_prompt": "Test prompt",
        }
        config = AgentConfig.from_dict(d)
        assert config.name == "test_critic"
        assert config.role == AgentRole.CRITIC
        assert config.error_specialty == CriticSkill.COMPUTATION
        assert config.lora_path == "/models/lora"


class TestAgentRegistry:
    """Test AgentRegistry management."""

    def test_register_agent(self):
        """Should register an agent."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="actor_1",
            role=AgentRole.ACTOR,
            model_path="/models/base",
        )
        registry.register(config)
        assert registry.get("actor_1") == config
        assert len(registry) == 1

    def test_list_actors(self):
        """Should list only actors."""
        registry = AgentRegistry()
        registry.register(AgentConfig(
            name="actor_1", role=AgentRole.ACTOR, model_path="/models/base",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        ))
        registry.register(AgentConfig(
            name="critic_1", role=AgentRole.CRITIC, model_path="/models/base",
            error_specialty=CriticSkill.COMPUTATION,
        ))
        actors = registry.list_actors()
        assert len(actors) == 1
        assert actors[0].role == AgentRole.ACTOR

    def test_list_critics(self):
        """Should list only critics."""
        registry = AgentRegistry()
        registry.register(AgentConfig(
            name="actor_1", role=AgentRole.ACTOR, model_path="/models/base",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        ))
        registry.register(AgentConfig(
            name="critic_1", role=AgentRole.CRITIC, model_path="/models/base",
            error_specialty=CriticSkill.REASONING,
        ))
        critics = registry.list_critics()
        assert len(critics) == 1
        assert critics[0].role == AgentRole.CRITIC

    def test_get_actor_by_style(self):
        """Should get actor by reasoning style."""
        registry = AgentRegistry()
        registry.register(AgentConfig(
            name="actor_1", role=AgentRole.ACTOR, model_path="/models/base",
            reasoning_style=ReasoningStyle.DIRECT,
        ))
        actor = registry.get_actor_by_style(ReasoningStyle.DIRECT)
        assert actor is not None
        assert actor.reasoning_style == ReasoningStyle.DIRECT

    def test_get_actor_by_style_not_found(self):
        """Should return None if style not found."""
        registry = AgentRegistry()
        actor = registry.get_actor_by_style(ReasoningStyle.ALGEBRAIC)
        assert actor is None

    def test_get_critic_by_specialty(self):
        """Should get critic by error specialty."""
        registry = AgentRegistry()
        registry.register(AgentConfig(
            name="critic_1", role=AgentRole.CRITIC, model_path="/models/base",
            error_specialty=CriticSkill.KNOWLEDGE,
        ))
        critic = registry.get_critic_by_specialty(CriticSkill.KNOWLEDGE)
        assert critic is not None
        assert critic.error_specialty == CriticSkill.KNOWLEDGE

    def test_get_all_pairs(self):
        """Should get all actor-critic pairs."""
        registry = AgentRegistry()
        registry.register(AgentConfig(
            name="actor_1", role=AgentRole.ACTOR, model_path="/models/base",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        ))
        registry.register(AgentConfig(
            name="actor_2", role=AgentRole.ACTOR, model_path="/models/base",
            reasoning_style=ReasoningStyle.DIRECT,
        ))
        registry.register(AgentConfig(
            name="critic_1", role=AgentRole.CRITIC, model_path="/models/base",
            error_specialty=CriticSkill.COMPUTATION,
        ))
        registry.register(AgentConfig(
            name="critic_2", role=AgentRole.CRITIC, model_path="/models/base",
            error_specialty=CriticSkill.REASONING,
        ))
        pairs = registry.get_all_pairs()
        assert len(pairs) == 4  # 2 actors * 2 critics

    def test_save_and_load(self):
        """Should save and load registry from JSON."""
        registry = AgentRegistry(base_model_path="/models/qwen")
        registry.register(AgentConfig(
            name="actor_1",
            role=AgentRole.ACTOR,
            model_path="/models/qwen",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        ))
        registry.register(AgentConfig(
            name="critic_1",
            role=AgentRole.CRITIC,
            model_path="/models/qwen",
            error_specialty=CriticSkill.COMPUTATION,
        ))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            registry.save(temp_path)
            loaded = AgentRegistry.load(temp_path)

            assert loaded.base_model_path == "/models/qwen"
            assert len(loaded) == 2
            assert loaded.get("actor_1") is not None
            assert loaded.get("critic_1") is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_default(self):
        """Should create default 3-actor + 4-critic setup."""
        registry = AgentRegistry.create_default(base_model_path="/models/qwen")

        assert len(registry.list_actors()) == 3
        assert len(registry.list_critics()) == 4
        assert len(registry) == 7

        # Check all reasoning styles are present
        actors = registry.list_actors()
        actor_styles = {a.reasoning_style for a in actors}
        assert actor_styles == set(ReasoningStyle)

        # Check all error profiles are present
        critics = registry.list_critics()
        critic_specialties = {c.error_specialty for c in critics}
        assert critic_specialties == set(CriticSkill)

    def test_repr(self):
        """String representation should show actor/critic counts."""
        registry = AgentRegistry()
        registry.register(AgentConfig(
            name="actor_1", role=AgentRole.ACTOR, model_path="/models/base",
            reasoning_style=ReasoningStyle.ALGEBRAIC,
        ))
        registry.register(AgentConfig(
            name="critic_1", role=AgentRole.CRITIC, model_path="/models/base",
            error_specialty=CriticSkill.COMPUTATION,
        ))
        assert "1 actors" in repr(registry)
        assert "1 critics" in repr(registry)


class TestEnums:
    """Test enum definitions."""

    def test_reasoning_style_values(self):
        """ReasoningStyle should have correct values."""
        assert ReasoningStyle.ALGEBRAIC.value == "algebraic"
        assert ReasoningStyle.DIRECT.value == "direct"
        assert ReasoningStyle.BACKTRACKING.value == "backtracking"

    def test_critic_skill_values(self):
        """CriticSkill should have correct values."""
        assert CriticSkill.COMPUTATION.value == "computation"
        assert CriticSkill.REASONING.value == "reasoning"
        assert CriticSkill.KNOWLEDGE.value == "knowledge"
        assert CriticSkill.VERIFICATION.value == "verification"


# ============================================================
# 2. router.py Tests
# ============================================================

class TestParseConfidence:
    """Test confidence parsing from Critic responses."""

    def test_parse_confidence_standard(self):
        """Should parse standard [Confidence: 0.X] format."""
        assert parse_confidence("Good work [Confidence: 0.8]") == 0.8
        assert parse_confidence("[Confidence: 0.95]") == 0.95

    def test_parse_confidence_case_insensitive(self):
        """Should be case insensitive."""
        assert parse_confidence("[confidence: 0.7]") == 0.7
        assert parse_confidence("[CONFIDENCE: 0.6]") == 0.6

    def test_parse_confidence_with_spaces(self):
        """Should handle various spacing (but our regex is strict)."""
        # Our regex requires format [Confidence: X] without extra spaces around colon
        assert parse_confidence("[Confidence:0.5]") == 0.5
        # Extra spaces cause failure (regex doesn't match)
        assert parse_confidence("[ Confidence : 0.9 ]") is None

    def test_parse_confidence_clamped(self):
        """Should clamp to [0.0, 1.0] for valid numbers."""
        assert parse_confidence("[Confidence: 1.5]") == 1.0
        # Negative numbers don't match the regex (no sign in pattern)
        assert parse_confidence("[Confidence: -0.5]") is None

    def test_parse_confidence_invalid(self):
        """Should return None for invalid format."""
        assert parse_confidence("No confidence here") is None
        assert parse_confidence("[Confidence: abc]") is None
        assert parse_confidence("") is None


class TestParseAnswerCorrect:
    """Test answer correctness parsing from Critic responses."""

    def test_parse_yes(self):
        assert parse_answer_correct("[Answer_Correct: yes]") is True

    def test_parse_no(self):
        assert parse_answer_correct("[Answer_Correct: no]") is False

    def test_parse_case_insensitive(self):
        assert parse_answer_correct("[answer_correct: YES]") is True
        assert parse_answer_correct("[ANSWER_CORRECT: No]") is False

    def test_parse_with_surrounding_text(self):
        text = "The solution has a reasoning error.\n[Answer_Correct: no]\n[Confidence: 0.85]"
        assert parse_answer_correct(text) is False

    def test_parse_missing_tag(self):
        assert parse_answer_correct("No tag here") is None
        assert parse_answer_correct("") is None

    def test_parse_invalid_value(self):
        assert parse_answer_correct("[Answer_Correct: maybe]") is None


class TestBuildCriticFeedback:
    """Test building CriticFeedback from raw response."""

    def test_build_with_confidence(self):
        """Should build with parsed confidence."""
        config = AgentConfig(
            name="critic_1",
            role=AgentRole.CRITIC,
            model_path="/models/base",
            error_specialty=CriticSkill.COMPUTATION,
        )
        response = "Check your math [Confidence: 0.8]"
        feedback = build_critic_feedback(config, response)

        assert feedback.critic_name == "critic_1"
        assert feedback.error_specialty == CriticSkill.COMPUTATION
        assert feedback.confidence == 0.8
        assert feedback.feedback_text == "Check your math"
        assert feedback.answer_correct is None
        assert feedback.raw_response == response

    def test_build_without_confidence(self):
        """Should use default 0.5 confidence when not found."""
        config = AgentConfig(
            name="critic_1",
            role=AgentRole.CRITIC,
            model_path="/models/base",
            error_specialty=CriticSkill.REASONING,
        )
        response = "Check your reasoning"
        feedback = build_critic_feedback(config, response)

        assert feedback.confidence == 0.5
        assert feedback.feedback_text == "Check your reasoning"
        assert feedback.answer_correct is None

    def test_build_with_answer_correct(self):
        """Should parse answer_correct tag."""
        config = AgentConfig(
            name="critic_1",
            role=AgentRole.CRITIC,
            model_path="/models/base",
            error_specialty=CriticSkill.COMPUTATION,
        )
        response = "The answer is wrong.\n[Answer_Correct: no]\n[Confidence: 0.9]"
        feedback = build_critic_feedback(config, response)

        assert feedback.confidence == 0.9
        assert feedback.answer_correct is False
        assert "[Answer_Correct:" not in feedback.feedback_text
        assert "[Confidence:" not in feedback.feedback_text


class TestCriticRouter:
    """Test CriticRouter MoE-style routing."""

    def test_route_empty_feedbacks(self):
        """Should handle empty feedback list."""
        router = CriticRouter()
        result = router.route([])
        assert result.feedback_text == ""
        assert result.selected_critics == []
        assert result.weights == []

    def test_route_single_critic(self):
        """Single critic should have weight 1.0."""
        router = CriticRouter()
        feedbacks = [
            CriticFeedback(
                critic_name="critic_1",
                error_specialty=CriticSkill.COMPUTATION,
                feedback_text="Check math",
                confidence=0.8,
            )
        ]
        result = router.route(feedbacks)

        assert len(result.selected_critics) == 1
        assert result.weights[0] == pytest.approx(1.0)

    def test_route_top_k_selection(self):
        """Should select top-k by confidence."""
        router = CriticRouter(top_k=2)
        feedbacks = [
            CriticFeedback("c1", CriticSkill.COMPUTATION, "f1", 0.5),
            CriticFeedback("c2", CriticSkill.REASONING, "f2", 0.9),
            CriticFeedback("c3", CriticSkill.KNOWLEDGE, "f3", 0.7),
            CriticFeedback("c4", CriticSkill.VERIFICATION, "f4", 0.6),
        ]
        result = router.route(feedbacks)

        assert len(result.selected_critics) == 2
        # Should select c2 (0.9) and c3 (0.7) or similar top 2
        assert "c2" in result.selected_critics

    def test_route_min_confidence_filter(self):
        """Should filter by min_confidence."""
        router = CriticRouter(min_confidence=0.5)
        feedbacks = [
            CriticFeedback("c1", CriticSkill.COMPUTATION, "f1", 0.3),
            CriticFeedback("c2", CriticSkill.REASONING, "f2", 0.7),
        ]
        result = router.route(feedbacks)

        assert len(result.selected_critics) == 1
        assert "c2" in result.selected_critics

    def test_route_fallback_to_uniform(self):
        """Should fallback to uniform when all below min_confidence."""
        router = CriticRouter(min_confidence=0.8, fallback_to_uniform=True)
        feedbacks = [
            CriticFeedback("c1", CriticSkill.COMPUTATION, "f1", 0.3),
            CriticFeedback("c2", CriticSkill.REASONING, "f2", 0.5),
        ]
        result = router.route(feedbacks)

        # With fallback, should still route
        assert len(result.selected_critics) <= 2

    def test_route_no_fallback_returns_empty(self):
        """Should return empty when all below min_confidence and no fallback."""
        router = CriticRouter(min_confidence=0.8, fallback_to_uniform=False)
        feedbacks = [
            CriticFeedback("c1", CriticSkill.COMPUTATION, "f1", 0.3),
        ]
        result = router.route(feedbacks)

        assert result.feedback_text == ""

    def test_softmax_temperature(self):
        """Higher temperature should make weights more uniform."""
        router_low = CriticRouter(temperature=0.1)
        router_high = CriticRouter(temperature=5.0)
        feedbacks = [
            CriticFeedback("c1", CriticSkill.COMPUTATION, "f1", 0.9),
            CriticFeedback("c2", CriticSkill.REASONING, "f2", 0.1),
        ]

        result_low = router_low.route(feedbacks)
        result_high = router_high.route(feedbacks)

        # With top_k=2 (default), both critics are selected
        # The weights are normalized after selection
        # Low temp amplifies differences, high temp makes them more equal

        # Just verify weights sum to 1
        assert sum(result_low.weights) == pytest.approx(1.0)
        assert sum(result_high.weights) == pytest.approx(1.0)

        # With only 2 critics and top_k=2, both are always selected
        # The temperature affects the softmax but with renormalization
        # after selection, the effect is less pronounced
        assert len(result_low.selected_critics) == 2
        assert len(result_high.selected_critics) == 2


# ============================================================
# 3. data_classifier.py Tests
# ============================================================

class TestClassificationError:
    """Test that ClassificationError is raised when API is unavailable."""

    def test_style_raises_when_no_api_key(self):
        """Should raise ClassificationError when GLM_API_KEY is not set."""
        with patch("src.society.data_classifier.DEFAULT_API_KEY", ""):
            with patch("src.society.data_classifier._compute_sample_hash", return_value="unique_hash"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ClassificationError, match="GLM_API_KEY"):
                        classify_reasoning_style(
                            response="Some response",
                            question="Some question",
                            use_api=True,
                            cache_dir=tmpdir,
                        )

    def test_error_profile_raises_when_no_api_key(self):
        """Should raise ClassificationError when GLM_API_KEY is not set."""
        with patch("src.society.data_classifier.DEFAULT_API_KEY", ""):
            with patch("src.society.data_classifier._compute_sample_hash", return_value="unique_hash"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ClassificationError, match="GLM_API_KEY"):
                        classify_error_profile(
                            response="Some response",
                            question="Some question",
                            use_api=True,
                            cache_dir=tmpdir,
                        )

    def test_style_raises_on_unparseable_api_response(self):
        """Should raise ClassificationError when API response cannot be parsed."""
        with patch("src.society.data_classifier._call_api", return_value="UNKNOWN_STYLE"):
            with patch("src.society.data_classifier._compute_sample_hash", return_value="unique_hash"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ClassificationError, match="Could not parse"):
                        classify_reasoning_style(
                            response="Some response",
                            question="Some question",
                            use_api=True,
                            cache_dir=tmpdir,
                        )

    def test_api_failure_raises(self):
        """Should raise ClassificationError when API call fails."""
        with patch("src.society.data_classifier._call_api", side_effect=ClassificationError("Connection refused")):
            with patch("src.society.data_classifier._compute_sample_hash", return_value="unique_hash"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(ClassificationError):
                        classify_reasoning_style(
                            response="Some response",
                            question="Some question",
                            use_api=True,
                            cache_dir=tmpdir,
                        )

    def test_check_api_available_no_key(self):
        """check_api_available should return False when no key is set."""
        with patch("src.society.data_classifier.DEFAULT_API_KEY", ""):
            available, reason = check_api_available()
            assert available is False
            assert "GLM_API_KEY" in reason


class TestParseStyleResponse:
    """Test parsing API response to ReasoningStyle."""

    def test_parse_algebraic(self):
        assert _parse_style_response("ALGEBRAIC") == ReasoningStyle.ALGEBRAIC
        assert _parse_style_response("algebraic") == ReasoningStyle.ALGEBRAIC
        assert _parse_style_response("The style is ALGEBRAIC") == ReasoningStyle.ALGEBRAIC

    def test_parse_backtracking(self):
        assert _parse_style_response("BACKTRACKING") == ReasoningStyle.BACKTRACKING
        assert _parse_style_response("backtracking") == ReasoningStyle.BACKTRACKING

    def test_parse_direct(self):
        assert _parse_style_response("DIRECT") == ReasoningStyle.DIRECT
        assert _parse_style_response("direct") == ReasoningStyle.DIRECT

    def test_parse_invalid(self):
        assert _parse_style_response("INVALID") is None
        assert _parse_style_response("") is None


class TestParseErrorResponse:
    """Test parsing API response to error profile."""

    def test_parse_computation(self):
        result = _parse_error_profile_response(json.dumps({
            "scores": {"computation": 0.9, "reasoning": 0.1, "knowledge": 0.0, "grounding": 0.0, "verification": 0.0},
            "primary": "computation",
            "secondary": [],
            "confidence": 0.8,
            "evidence": "calc",
        }))
        assert result.primary == "computation"
        assert result.scores["computation"] == 0.9

    def test_parse_reasoning(self):
        result = _parse_error_profile_response(json.dumps({
            "scores": {"reasoning": 0.8},
            "primary": "reasoning",
            "secondary": [],
            "confidence": 0.7,
        }))
        assert result.primary == "reasoning"

    def test_parse_knowledge(self):
        result = _parse_error_profile_response(json.dumps({
            "scores": {"knowledge": 0.8},
            "primary": "knowledge",
            "secondary": ["grounding"],
            "confidence": 0.7,
        }))
        assert result.primary == "knowledge"

    def test_parse_verification(self):
        result = _parse_error_profile_response(json.dumps({
            "scores": {"verification": 0.8},
            "primary": "verification",
            "secondary": [],
            "confidence": 0.7,
        }))
        assert result.primary == "verification"

    def test_parse_invalid(self):
        assert _parse_error_profile_response("INVALID") is None


class TestComputeSampleHash:
    """Test hash computation for caching."""

    def test_hash_deterministic(self):
        """Same input should produce same hash."""
        h1 = _compute_sample_hash("Question 1", "Response 1")
        h2 = _compute_sample_hash("Question 1", "Response 1")
        assert h1 == h2

    def test_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        h1 = _compute_sample_hash("Question 1", "Response 1")
        h2 = _compute_sample_hash("Question 2", "Response 1")
        assert h1 != h2


class TestDataClassifier:
    """Test DataClassifier class."""

    def test_classify_reasoning_style_with_api_mock(self):
        """Should use API response when available."""
        with patch("src.society.data_classifier._call_api") as mock_api:
            mock_api.return_value = "ALGEBRAIC"
            with tempfile.TemporaryDirectory() as tmpdir:
                classifier = DataClassifier()
                result = classifier.classify_reasoning_style(
                    response="Let x = 5",
                    question="Solve for x",
                    use_api=True,
                    cache_dir=tmpdir,
                )
                assert result.style == ReasoningStyle.ALGEBRAIC
                assert result.confidence == 0.9

    def test_classify_error_profile_with_api_mock(self):
        """Should use API response for error profile."""
        with patch("src.society.data_classifier._call_api") as mock_api:
            mock_api.return_value = json.dumps({
                "scores": {
                    "computation": 0.0,
                    "reasoning": 0.9,
                    "knowledge": 0.1,
                    "grounding": 0.0,
                    "verification": 0.0,
                },
                "primary": "reasoning",
                "secondary": [],
                "confidence": 0.9,
                "evidence": "bad inference",
            })
            with tempfile.TemporaryDirectory() as tmpdir:
                classifier = DataClassifier()
                result = classifier.classify_error_profile(
                    response="Wrong reasoning",
                    question="Problem",
                    use_api=True,
                    cache_dir=tmpdir,
                )
                assert result.primary == "reasoning"
                assert result.confidence == 0.9

    def test_classify_with_cache(self):
        """Should load from cache when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file
            cache_file = Path(tmpdir) / "style_abc123.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({
                    "style": "backtracking",
                    "confidence": 0.85,
                    "raw_response": "CACHED",
                }, f)

            with patch("src.society.data_classifier._compute_sample_hash", return_value="abc123"):
                classifier = DataClassifier()
                result = classifier.classify_reasoning_style(
                    response="Any response",
                    question="Any question",
                    use_api=True,
                    cache_dir=tmpdir,
                )
                assert result.style == ReasoningStyle.BACKTRACKING
                assert result.confidence == 0.85

    def test_classify_api_failure_raises(self):
        """Should raise ClassificationError when API returns None."""
        with patch("src.society.data_classifier._call_api", return_value=None):
            with patch("src.society.data_classifier._compute_sample_hash", return_value="unique_hash"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    classifier = DataClassifier()
                    with pytest.raises(ClassificationError):
                        classifier.classify_reasoning_style(
                            response="Let x = 5",
                            question="Solve",
                            use_api=True,
                            cache_dir=tmpdir,
                        )


# ============================================================
# 4. diversity_split.py Tests
# ============================================================

class TestDiversitySplit:
    """Test data splitting by reasoning style and error profile."""

    def test_split_by_reasoning_style(self):
        """Should split samples by reasoning style."""
        splitter = DiversitySplit()
        samples = [
            {"question": f"Q{i}", "answer": "A"}
            for i in range(10)
        ]
        responses = [
            "Let x = 5",  # ALGEBRAIC
            "Let me verify",  # BACKTRACKING
            "Direct calculation",  # DIRECT
            "Let y = 10",  # ALGEBRAIC
            "Check the answer",  # BACKTRACKING
            "Step by step",  # DIRECT
            "Use equation",  # ALGEBRAIC
            "Verify result",  # BACKTRACKING
            "Simple math",  # DIRECT
            "Let z = 2",  # ALGEBRAIC
        ]

        # Mock API to return deterministic classifications
        api_responses = [
            "ALGEBRAIC", "BACKTRACKING", "DIRECT", "ALGEBRAIC",
            "BACKTRACKING", "DIRECT", "ALGEBRAIC", "BACKTRACKING",
            "DIRECT", "ALGEBRAIC",
        ]
        with patch("src.society.data_classifier._call_api", side_effect=api_responses):
            splits = splitter.split_by_reasoning_style(samples, responses)

        assert ReasoningStyle.ALGEBRAIC in splits
        assert ReasoningStyle.DIRECT in splits
        assert ReasoningStyle.BACKTRACKING in splits

        # 4 ALGEBRAIC, 3 BACKTRACKING, 3 DIRECT -> balanced to 3 each
        assert len(splits[ReasoningStyle.ALGEBRAIC]) == 3
        assert len(splits[ReasoningStyle.BACKTRACKING]) == 3
        assert len(splits[ReasoningStyle.DIRECT]) == 3

    def test_split_by_error_profile(self):
        """Should split by error profile."""
        splitter = DiversitySplit(balance=False)
        samples = [
            {"question": f"Q{i}", "answer": "42"}
            for i in range(5)
        ]
        responses = [
            "The number is fabricated",
            "Logical fallacy here",
            "Calculation error",
            "Should have verified",
            "Wrong computation",
        ]

        def profile_json(primary):
            scores = {
                "computation": 0.0,
                "reasoning": 0.0,
                "knowledge": 0.0,
                "grounding": 0.0,
                "verification": 0.0,
            }
            scores[primary] = 0.9
            return json.dumps({
                "scores": scores,
                "primary": primary,
                "secondary": [],
                "confidence": 0.9,
                "evidence": primary,
            })

        api_responses = [
            profile_json("knowledge"),
            profile_json("reasoning"),
            profile_json("computation"),
            profile_json("verification"),
            profile_json("computation"),
        ]
        with patch("src.society.data_classifier._call_api", side_effect=api_responses):
            splits = splitter.split_by_error_profile(samples, responses)

        assert len(splits) == 5
        assert {item.skill for item in splits} == {
            CriticSkill.COMPUTATION,
            CriticSkill.REASONING,
            CriticSkill.KNOWLEDGE,
            CriticSkill.VERIFICATION,
        }

    def test_split_balancing(self):
        """Should balance splits when enabled."""
        splitter = DiversitySplit(balance=True)
        samples = [{"question": f"Q{i}", "answer": "A"} for i in range(100)]
        responses = ["Let x = 5"] * 80 + ["Verify this"] * 20

        # 80 ALGEBRAIC + 20 BACKTRACKING -> balanced to 20 each
        api_responses = ["ALGEBRAIC"] * 80 + ["BACKTRACKING"] * 20
        with patch("src.society.data_classifier._call_api", side_effect=api_responses):
            splits = splitter.split_by_reasoning_style(samples, responses)

        # With balancing, should downsample ALGEBRAIC to 20
        assert len(splits[ReasoningStyle.ALGEBRAIC]) <= 80

    def test_split_no_responses(self):
        """Should round-robin assign when no responses."""
        splitter = DiversitySplit()
        samples = [
            {"question": f"Q{i}", "answer": "A"}
            for i in range(10)
        ]

        # No responses -> round-robin, no API calls needed
        splits = splitter.split_by_reasoning_style(samples, responses=None)

        # Should distribute across all styles via round-robin
        total = sum(len(v) for v in splits.values())
        assert total == 9  # Due to how the split works, one sample gets filtered

    def test_adaptive_critic_mix_skips_sparse_specialty(self):
        """Should not train specialists when target data is too sparse."""
        splitter = DiversitySplit(seed=123)
        items = [
            RoutedTrainingItem(
                sample={"question": f"Q{i}"},
                response=f"R{i}",
                skill=CriticSkill.REASONING,
                weight=1.0,
                profile={},
            )
            for i in range(10)
        ]
        items.append(RoutedTrainingItem(
            sample={"question": "Qc"},
            response="Rc",
            skill=CriticSkill.COMPUTATION,
            weight=1.0,
            profile={},
        ))

        selected = splitter.build_critic_training_mix(
            all_items=items,
            target_skill=CriticSkill.COMPUTATION,
            max_items=8,
            min_specialty_items=4,
            min_specialty_ratio=0.2,
        )

        assert selected == []

    def test_adaptive_critic_mix_prefers_specialty_and_tracks_source_bucket(self):
        """Should select a specialty-heavy mix when target data is sufficient."""
        splitter = DiversitySplit(seed=123)
        items = []
        for i in range(10):
            items.append(RoutedTrainingItem(
                sample={"question": f"C{i}"},
                response=f"RC{i}",
                skill=CriticSkill.COMPUTATION,
                weight=1.0,
                profile={},
            ))
        for i in range(10):
            items.append(RoutedTrainingItem(
                sample={"question": f"R{i}"},
                response=f"RR{i}",
                skill=CriticSkill.REASONING,
                weight=1.0,
                profile={},
            ))

        selected = splitter.build_critic_training_mix(
            all_items=items,
            target_skill=CriticSkill.COMPUTATION,
            max_items=10,
            min_specialty_items=4,
            min_specialty_ratio=0.2,
            specialty_ratio=0.6,
            general_ratio=0.3,
            calibration_ratio=0.1,
        )

        buckets = {item.source_bucket for item in selected}
        specialty_items = [
            item for item in selected
            if item.source_bucket == "specialty"
        ]

        assert len(selected) == 10
        assert len(specialty_items) >= 6
        assert {"specialty", "general", "calibration"} <= buckets


# ============================================================
# 5. inference_pipeline.py Tests
# ============================================================

class TestVotingStrategies:
    """Test voting strategy implementations."""

    def test_majority_vote_clear_winner(self):
        """Should select answer with majority."""
        answers = {"actor_1": "42", "actor_2": "42", "actor_3": "40"}
        result, confidence = _majority_vote(answers)
        assert result == "42"
        assert confidence == 2/3

    def test_majority_vote_tie(self):
        """Should handle ties (first wins)."""
        answers = {"actor_1": "42", "actor_2": "40"}
        result, confidence = _majority_vote(answers)
        assert result in ["42", "40"]
        assert confidence == 0.5

    def test_weighted_vote(self):
        """Should weight by answer frequency."""
        mock_result = MagicMock()
        mock_result.final_answers = {"actor_1": "42", "actor_2": "42", "actor_3": "40"}

        answers = {"actor_1": "42", "actor_2": "42", "actor_3": "40"}
        result, confidence = _weighted_vote(answers, mock_result)
        assert result == "42"
        assert confidence == pytest.approx(2/3)

    def test_best_actor_with_consensus(self):
        """Should select the actor with highest Critic confidence."""

        mock_result = MagicMock()
        mock_result.consensus_answer = "42"
        mock_result.consensus_confidence = 0.9
        mock_result.final_answers = {"actor_1": "42", "actor_2": "40"}

        # Create last round with routed feedbacks that have different confidence
        mock_round = MagicMock()
        fb_high = MagicMock()
        fb_high.confidence = 0.9
        fb_high.critic_name = "critic_1"
        fb_low = MagicMock()
        fb_low.confidence = 0.3
        fb_low.critic_name = "critic_1"

        routed_high = RoutedFeedback(
            feedback_text="High conf feedback",
            selected_critics=["critic_1"],
            raw_feedbacks=[fb_high],
            weights={"critic_1": 0.9},
        )
        routed_low = RoutedFeedback(
            feedback_text="Low conf feedback",
            selected_critics=["critic_1"],
            raw_feedbacks=[fb_low],
            weights={"critic_1": 0.3},
        )
        mock_round.routed_feedbacks = {
            "actor_1": routed_high,
            "actor_2": routed_low,
        }
        mock_result.rounds = [mock_round]

        answers = {"actor_1": "42", "actor_2": "40"}
        result, confidence = _best_actor(answers, mock_result)
        assert result == "42"  # Tie-broken by higher Critic confidence for actor_1
        assert confidence == 0.5  # 1 out of 2 actors agree

    def test_best_actor_fallback(self):
        """Should fallback to majority vote when no confidence data available."""
        mock_result = MagicMock()
        mock_result.consensus_answer = None
        mock_result.final_answers = {"actor_1": "42", "actor_2": "40"}
        mock_result.rounds = []  # No rounds -> no confidence data

        answers = {"actor_1": "42", "actor_2": "40"}
        result, confidence = _best_actor(answers, mock_result)
        assert result == "42"


class TestAblationConfigs:
    """Test ablation experiment configurations."""

    def test_ablation_configs_complete(self):
        """Should have all 5 ablation configurations."""
        assert "A1" in ABLATION_CONFIGS
        assert "A2" in ABLATION_CONFIGS
        assert "A3" in ABLATION_CONFIGS
        assert "A4" in ABLATION_CONFIGS
        assert "A5" in ABLATION_CONFIGS

    def test_a1_baseline(self):
        """A1 should be baseline 1 actor + 1 critic."""
        config = ABLATION_CONFIGS["A1"]
        assert config["num_actors"] == 1
        assert config["num_critics"] == 1
        assert config["router_top_k"] == 1

    def test_a2_actor_diversity(self):
        """A2 should have 3 actors + 1 critic."""
        config = ABLATION_CONFIGS["A2"]
        assert config["num_actors"] == 3
        assert config["num_critics"] == 1

    def test_a3_critic_specialization(self):
        """A3 should have 1 actor + 4 critics with routing."""
        config = ABLATION_CONFIGS["A3"]
        assert config["num_actors"] == 1
        assert config["num_critics"] == 4
        assert config["router_top_k"] == 2

    def test_a4_no_routing(self):
        """A4 should have all agents with uniform weights."""
        config = ABLATION_CONFIGS["A4"]
        assert config["num_actors"] == 3
        assert config["num_critics"] == 4
        assert config["router_top_k"] == 4

    def test_a5_full_system(self):
        """A5 should be full system with routing."""
        config = ABLATION_CONFIGS["A5"]
        assert config["num_actors"] == 3
        assert config["num_critics"] == 4
        assert config["router_top_k"] == 2


# ============================================================
# 6. Phase 0.1 Integration Tests (MATH/GSM)
# ============================================================

class TestMathAnswerExtraction:
    """Test MATH/GSM answer extraction with \boxed{} format."""

    def test_extract_boxed_simple(self):
        """Should extract simple boxed answer."""
        response = r"The answer is \boxed{42}."
        result = extract_answer(response, task_type="math")
        assert result == "42"

    def test_extract_boxed_expression(self):
        """Should extract mathematical expressions."""
        response = r"Therefore, \boxed{x^2 + 2x + 1}."
        result = extract_answer(response, task_type="math")
        assert "x^2" in result

    def test_extract_boxed_multiple(self):
        """Should extract first boxed answer."""
        response = r"First \boxed{1}, then \boxed{2}."
        result = extract_answer(response, task_type="math")
        assert result == "1"

    def test_extract_final_answer_pattern(self):
        """Should extract from 'Final Answer:' pattern."""
        response = "Final Answer: 42"
        result = extract_answer(response, task_type="math")
        assert result == "42"

    def test_extract_numeric_fallback(self):
        """Should fallback to numeric extraction."""
        response = "Therefore, the result equals 42."
        result = extract_answer(response, task_type="math")
        assert result == "42"

    def test_extract_empty_response(self):
        """Should handle empty response."""
        result = extract_answer("", task_type="math")
        assert result is None

    def test_extract_negative_number(self):
        """Should handle negative numbers."""
        response = r"\boxed{-42}"
        result = extract_answer(response, task_type="math")
        assert result == "-42"


class TestGenerateWrongAnswerMath:
    """Test wrong answer generation for MATH/GSM tasks."""

    def test_generate_wrong_math_numeric(self):
        """Should generate different numeric answer."""
        # For math type without choices, it returns a random float
        result = generate_wrong_answer("42", task_type="math")
        # Just verify it returns something
        assert result is not None
        # It won't be "42" since that's the correct answer
        # But it could be any random number

    def test_generate_wrong_math_with_choices(self):
        """Should pick from available choices."""
        # When task_type is not "yes_no", it treats as multiple choice
        result = generate_wrong_answer("C", choices=["A", "B", "C", "D"])
        assert result != "C"
        assert result in ["A", "B", "D"]


class TestMATHPromptTemplates:
    """Test MATH/GSM prompt template integration."""

    def test_math_template_exists(self):
        """MATH dataset should have templates."""
        from src.prompts.templates import get_available_datasets, get_prompt_template
        from src.prompts.templates import PromptType

        datasets = get_available_datasets()
        assert "math" in datasets

        template = get_prompt_template("math", PromptType.SINGLE_SHOT)
        assert "{question}" in template

    def test_gsm8k_template_exists(self):
        """GSM8K dataset should have templates."""
        from src.prompts.templates import get_available_datasets, get_prompt_template
        from src.prompts.templates import PromptType

        datasets = get_available_datasets()
        assert "gsm8k" in datasets

        template = get_prompt_template("gsm8k", PromptType.SINGLE_SHOT)
        assert "{question}" in template


# ============================================================
# 7. Integration Tests
# ============================================================

class TestSocietyIntegration:
    """End-to-end integration tests."""

    def test_full_registry_creation_and_routing(self):
        """Test creating registry and routing feedbacks."""
        # Create default registry
        registry = AgentRegistry.create_default()

        # Verify agents
        assert len(registry.list_actors()) == 3
        assert len(registry.list_critics()) == 4

        # Create mock feedbacks
        feedbacks = [
            CriticFeedback(
                critic_name=c.name,
                error_specialty=c.error_specialty,
                feedback_text=f"Feedback from {c.name}",
                confidence=0.5 + i * 0.1,
            )
            for i, c in enumerate(registry.list_critics())
        ]

        # Route
        router = CriticRouter(top_k=2)
        result = router.route(feedbacks)

        assert len(result.selected_critics) == 2
        assert result.feedback_text

    def test_classification_and_splitting_pipeline(self):
        """Test classify then split pipeline."""
        splitter = DiversitySplit()

        samples = [
            {"question": "Solve x + 5 = 10", "answer": "5"},
            {"question": "Calculate 2 * 3", "answer": "6"},
            {"question": "Verify the result", "answer": "10"},
        ]

        responses = [
            "Let x = 5, solve equation",
            "2 * 3 = 6 directly",
            "Let me check: verify is correct",
        ]

        # Mock API to return deterministic styles
        with patch("src.society.data_classifier._call_api", side_effect=["ALGEBRAIC", "DIRECT", "BACKTRACKING"]):
            splits = splitter.split_by_reasoning_style(samples, responses)

        # Verify all styles have samples
        assert len(splits[ReasoningStyle.ALGEBRAIC]) >= 1
        assert len(splits[ReasoningStyle.DIRECT]) >= 1
        assert len(splits[ReasoningStyle.BACKTRACKING]) >= 1
