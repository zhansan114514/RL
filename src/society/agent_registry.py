"""
Agent Registry for managing diverse Actor-Critic Society.

Defines agent types with specialized roles:
- 3 Actors with MMLU-native reasoning styles (DIRECT, EVIDENCE, ELIMINATION)
- 5 Critics with critic-skill specialties (COMPUTATION, REASONING, KNOWLEDGE, GROUNDING, VERIFICATION)

Each Agent has its own LoRA adapter on top of a shared base model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Enums for agent specialization
# ============================================================

class AgentRole(Enum):
    """Role in the Actor-Critic framework."""
    ACTOR = "actor"
    CRITIC = "critic"


class ReasoningStyle(Enum):
    """
    Actor reasoning styles for MMLU-style multiple-choice tasks.
    """
    DIRECT = "direct"              # Concise answer with minimal explanation
    EVIDENCE = "evidence"          # Explicit facts, concepts, definitions, wording
    ELIMINATION = "elimination"    # Compare options and rule out alternatives


class CriticSkill(Enum):
    """
    Critic skill specialties for targeted feedback.

    These skills are dataset-independent.  They avoid using a single "logic"
    label as the catch-all bucket for broad knowledge and multiple-choice tasks.
    """
    COMPUTATION = "computation"        # Calculation, symbolic manipulation, formula computation
    REASONING = "reasoning"            # Inference chain, rule application, causal/logical jumps
    KNOWLEDGE = "knowledge"            # Factual/domain knowledge and concepts
    GROUNDING = "grounding"            # Support from question, choices, passage, or context
    VERIFICATION = "verification"      # Final answer checks, option mapping, self-check failures


# ============================================================
# Robust enum resolution (no silent fallback to defaults)
# ============================================================

def resolve_reasoning_style(value: str) -> ReasoningStyle:
    """Resolve a string to ReasoningStyle with robust case-insensitive matching.

    Matching priority:
      1. Exact value match  (e.g. "evidence")
      2. Exact name match   (e.g. "EVIDENCE")
      3. Case-insensitive value match (e.g. "Evidence")

    Raises ValueError (never silently falls back to a default).
    """
    if not value:
        raise ValueError("Cannot resolve empty string to ReasoningStyle")

    # 1. Exact value match
    try:
        return ReasoningStyle(value)
    except ValueError:
        pass

    # 2. Exact name match
    try:
        return ReasoningStyle[value]
    except KeyError:
        pass

    # 3. Case-insensitive value match
    lower = value.lower()
    for style in ReasoningStyle:
        if style.value == lower:
            return style

    raise ValueError(
        f"Cannot resolve '{value}' to ReasoningStyle. "
        f"Valid values: {[s.value for s in ReasoningStyle]}, "
        f"valid names: {[s.name for s in ReasoningStyle]}"
    )


def resolve_critic_skill(value: str) -> CriticSkill:
    """Resolve a string to CriticSkill with robust case-insensitive matching.

    Matching priority:
      1. Exact value match  (e.g. "computation")
      2. Exact name match   (e.g. "COMPUTATION")
      3. Case-insensitive value match (e.g. "Computation")

    Raises ValueError (never silently falls back to a default).
    """
    if not value:
        raise ValueError("Cannot resolve empty string to CriticSkill")

    # 1. Exact value match
    try:
        return CriticSkill(value)
    except ValueError:
        pass

    # 2. Exact name match
    try:
        return CriticSkill[value]
    except KeyError:
        pass

    # 3. Case-insensitive value match
    lower = value.lower()
    aliases = {
        "calculation": CriticSkill.COMPUTATION,
        "calculus": CriticSkill.COMPUTATION,
        "factual": CriticSkill.KNOWLEDGE,
    }
    if lower in aliases:
        return aliases[lower]
    for skill in CriticSkill:
        if skill.value == lower:
            return skill

    raise ValueError(
        f"Cannot resolve '{value}' to CriticSkill. "
        f"Valid values: {[s.value for s in CriticSkill]}, "
        f"valid names: {[s.name for s in CriticSkill]}"
    )


# ============================================================
# Default prompt templates per specialization
# ============================================================

ACTOR_STYLE_PROMPTS = {
    ReasoningStyle.DIRECT: (
        "Solve with the shortest sufficient reasoning. Avoid unnecessary discussion "
        "and focus on the most direct route to the answer."
    ),
    ReasoningStyle.EVIDENCE: (
        "Identify key facts, concepts, definitions, wording, or evidence from "
        "the question and use them to justify the answer."
    ),
    ReasoningStyle.ELIMINATION: (
        "Compare answer choices, eliminate incorrect or weaker options, and "
        "explain why the selected option is best."
    ),
}

CRITIC_SPECIALTY_PROMPTS = {
    CriticSkill.COMPUTATION: (
        "You are a critic specializing in computation errors. "
        "Carefully check numerical calculations, algebra, symbolic manipulation, "
        "formula use, sign errors, units, and quantity-scale mistakes."
    ),
    CriticSkill.REASONING: (
        "You are a critic specializing in reasoning errors. "
        "Check the response for invalid inferences, missing steps, causal mistakes, "
        "wrong rule application, and unsupported reasoning jumps."
    ),
    CriticSkill.KNOWLEDGE: (
        "You are a critic specializing in knowledge errors. "
        "Check for wrong facts, domain concept confusion, missing background "
        "knowledge, and misuse of terminology."
    ),
    CriticSkill.GROUNDING: (
        "You are a critic specializing in grounding errors. "
        "Check whether the response is supported by the question, choices, passage, "
        "or provided context. Identify unsupported assumptions, ignored constraints, "
        "and contradictions with the given information."
    ),
    CriticSkill.VERIFICATION: (
        "You are a critic specializing in verification failures. "
        "Check final-answer consistency, answer-option mapping, extracted answers, "
        "and whether self-checks should have caught contradictions."
    ),
}

# ============================================================
# Agent configuration
# ============================================================

@dataclass
class AgentConfig:
    """Configuration for a single agent (Actor or Critic)."""

    name: str
    role: AgentRole
    model_path: str  # Base model path (shared)
    lora_path: Optional[str] = None  # Path to LoRA adapter (None = base model only)

    # Specialization (one per role type)
    reasoning_style: Optional[ReasoningStyle] = None  # For Actors
    error_specialty: Optional[CriticSkill] = None  # For Critics

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 512

    # System prompt (auto-generated if empty)
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build default system prompt based on specialization."""
        if self.role == AgentRole.ACTOR and self.reasoning_style:
            return ACTOR_STYLE_PROMPTS.get(self.reasoning_style, "")
        elif self.role == AgentRole.CRITIC and self.error_specialty:
            return CRITIC_SPECIALTY_PROMPTS.get(self.error_specialty, "")
        return ""

    @property
    def display_name(self) -> str:
        """Human-readable name with specialization."""
        if self.role == AgentRole.ACTOR and self.reasoning_style:
            return f"Actor-{self.reasoning_style.value}"
        elif self.role == AgentRole.CRITIC and self.error_specialty:
            return f"Critic-{self.error_specialty.value}"
        return self.name

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = {
            "name": self.name,
            "role": self.role.value,
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "reasoning_style": self.reasoning_style.value if self.reasoning_style else None,
            "error_specialty": self.error_specialty.value if self.error_specialty else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        """Deserialize from dictionary."""
        d = dict(d)
        d["role"] = AgentRole(d["role"])
        if d.get("reasoning_style"):
            d["reasoning_style"] = ReasoningStyle(d["reasoning_style"])
        if d.get("error_specialty"):
            d["error_specialty"] = resolve_critic_skill(d["error_specialty"])
        return cls(**d)


# ============================================================
# Agent Registry
# ============================================================

class AgentRegistry:
    """
    Registry for managing the Actor-Critic Society's agents.

    Tracks 3 Actors (DIRECT, EVIDENCE, ELIMINATION) and
    5 Critics (COMPUTATION, REASONING, KNOWLEDGE, GROUNDING, VERIFICATION),
    each with their own LoRA adapter path.
    """

    def __init__(self, base_model_path: str = ""):
        self._agents: dict[str, AgentConfig] = {}
        self.base_model_path = base_model_path

    def register(self, config: AgentConfig) -> None:
        """Register an agent."""
        self._agents[config.name] = config
        logger.info(f"Registered {config.display_name} ({config.name})")

    def get(self, name: str) -> Optional[AgentConfig]:
        """Get agent by name."""
        return self._agents.get(name)

    def list_actors(self) -> list[AgentConfig]:
        """List all registered Actors."""
        return [a for a in self._agents.values() if a.role == AgentRole.ACTOR]

    def list_critics(self) -> list[AgentConfig]:
        """List all registered Critics."""
        return [a for a in self._agents.values() if a.role == AgentRole.CRITIC]

    def get_actor_by_style(self, style: ReasoningStyle) -> Optional[AgentConfig]:
        """Get the Actor with a specific reasoning style."""
        for a in self._agents.values():
            if a.role == AgentRole.ACTOR and a.reasoning_style == style:
                return a
        return None

    def get_critic_by_specialty(self, specialty: CriticSkill) -> Optional[AgentConfig]:
        """Get the Critic with a specific skill specialty."""
        for a in self._agents.values():
            if a.role == AgentRole.CRITIC and a.error_specialty == specialty:
                return a
        return None

    def get_all_pairs(self) -> list[tuple[AgentConfig, AgentConfig]]:
        """Get all Actor-Critic pairs (for trajectory generation)."""
        actors = self.list_actors()
        critics = self.list_critics()
        return [(a, c) for a in actors for c in critics]

    def save(self, path: str | Path) -> None:
        """Save registry to JSON file (atomic write via tmp+rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "base_model_path": self.base_model_path,
            "agents": {name: cfg.to_dict() for name, cfg in self._agents.items()},
        }

        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.rename(path)
        logger.info(f"Registry saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AgentRegistry":
        """Load registry from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        registry = cls(base_model_path=data.get("base_model_path", ""))
        for name, cfg_dict in data.get("agents", {}).items():
            cfg = AgentConfig.from_dict(cfg_dict)
            registry.register(cfg)

        logger.info(f"Loaded {len(registry._agents)} agents from {path}")
        return registry

    @classmethod
    def create_default(
        cls,
        base_model_path: str = "Qwen/Qwen3-14B",
        cache_dir: str = "cache/society",
    ) -> "AgentRegistry":
        """Create a registry with the default 3-Actor + 5-Critic setup."""
        registry = cls(base_model_path=base_model_path)

        # 3 Actors with distinct reasoning styles
        for style in ReasoningStyle:
            agent_name = f"actor_{style.value}"
            lora_path = f"{cache_dir}/actors/{agent_name}/adapter"
            registry.register(AgentConfig(
                name=agent_name,
                role=AgentRole.ACTOR,
                model_path=base_model_path,
                lora_path=lora_path,
                reasoning_style=style,
            ))

        # 5 Critics with distinct skill specialties
        for specialty in CriticSkill:
            agent_name = f"critic_{specialty.value}"
            lora_path = f"{cache_dir}/critics/{agent_name}/adapter"
            registry.register(AgentConfig(
                name=agent_name,
                role=AgentRole.CRITIC,
                model_path=base_model_path,
                lora_path=lora_path,
                error_specialty=specialty,
            ))

        return registry

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        actors = len(self.list_actors())
        critics = len(self.list_critics())
        return f"AgentRegistry({actors} actors, {critics} critics)"
