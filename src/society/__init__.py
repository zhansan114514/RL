"""
Diverse Actor-Critic Society module.

Implements multi-agent deliberation with diverse thinking chains,
combining ACC-Collab (Actor-Critic collaboration) with Multiagent FT
(diverse thinking chains for complex reasoning).

Architecture:
- 3 Actors: ALGEBRAIC, DIRECT, BACKTRACKING reasoning styles
- 4 Critics: ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION error specialties
- MoE Router: Confidence-based softmax routing (no trainable params)
- Single GPU: Sequential LoRA load/unload with disk-based crash recovery
"""

from __future__ import annotations

from src.society.agent_registry import (
    AgentConfig,
    AgentRegistry,
    AgentRole,
    ReasoningStyle,
    ErrorType,
    ACTOR_STYLE_PROMPTS,
    CRITIC_SPECIALTY_PROMPTS,
)
from src.society.router import (
    CriticRouter,
    CriticFeedback,
    RoutedFeedback,
    parse_confidence,
    build_critic_feedback,
)
from src.society.multi_deliberation import (
    multi_agent_deliberate,
    multi_agent_deliberate_single_gpu,
    MultiDeliberationResult,
    DeliberationRound,
)
from src.society.data_classifier import (
    DataClassifier,
    ClassificationError,
    classify_reasoning_style,
    classify_error_type,
    check_api_available,
)
from src.society.diversity_split import DiversitySplit
from src.society.society_trainer import (
    society_alternating_train,
    SocietyTrainingResult,
)
from src.society.inference_pipeline import (
    society_inference,
    InferenceResult,
    run_ablation,
    ABLATION_CONFIGS,
)

__all__ = [
    # Agent Registry
    "AgentConfig",
    "AgentRegistry",
    "AgentRole",
    "ReasoningStyle",
    "ErrorType",
    "ACTOR_STYLE_PROMPTS",
    "CRITIC_SPECIALTY_PROMPTS",
    # Router
    "CriticRouter",
    "CriticFeedback",
    "RoutedFeedback",
    "parse_confidence",
    "build_critic_feedback",
    # Multi-Deliberation
    "multi_agent_deliberate",
    "multi_agent_deliberate_single_gpu",
    "MultiDeliberationResult",
    "DeliberationRound",
    # Data Classifier
    "DataClassifier",
    "ClassificationError",
    "classify_reasoning_style",
    "classify_error_type",
    "check_api_available",
    # Diversity Split
    "DiversitySplit",
    # Society Trainer
    "society_alternating_train",
    "SocietyTrainingResult",
    # Inference Pipeline
    "society_inference",
    "InferenceResult",
    "run_ablation",
    "ABLATION_CONFIGS",
]
