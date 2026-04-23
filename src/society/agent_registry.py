"""
Agent Registry for managing diverse Actor-Critic Society.

Defines agent types with specialized roles:
- 3 Actors with different reasoning styles (ALGEBRAIC, DIRECT, BACKTRACKING)
- 4 Critics with different error-type specialties (ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION)

Each Agent has its own LoRA adapter on top of a shared base model (Qwen2.5-7B-Instruct).
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
    Actor reasoning styles for diverse thinking chains.

    From experiment plan: each Actor specializes in a distinct problem-solving approach.
    """
    ALGEBRAIC = "algebraic"        # Symbolic manipulation, equations, variables
    DIRECT = "direct"              # Direct step-by-step numerical computation
    BACKTRACKING = "backtracking"  # Try-verify-revise approach


class ErrorType(Enum):
    """
    Critic error-type specialties for targeted feedback.

    From experiment plan: each Critic specializes in detecting and correcting
    a specific category of errors.
    """
    ARITHMETIC = "arithmetic"          # Correct reasoning but numerical calculation mistake
    LOGIC = "logic"                    # Flawed reasoning chain, wrong formula
    HALLUCINATION = "hallucination"    # Fabricated numbers, wrong theorem
    VERIFICATION = "verification"      # Failed self-check, missed the error


# ============================================================
# Robust enum resolution (no silent fallback to defaults)
# ============================================================

def resolve_reasoning_style(value: str) -> ReasoningStyle:
    """Resolve a string to ReasoningStyle with robust case-insensitive matching.

    Matching priority:
      1. Exact value match  (e.g. "algebraic")
      2. Exact name match   (e.g. "ALGEBRAIC")
      3. Case-insensitive value match (e.g. "Algebraic")

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


def resolve_error_type(value: str) -> ErrorType:
    """Resolve a string to ErrorType with robust case-insensitive matching.

    Matching priority:
      1. Exact value match  (e.g. "arithmetic")
      2. Exact name match   (e.g. "ARITHMETIC")
      3. Case-insensitive value match (e.g. "Arithmetic")

    Raises ValueError (never silently falls back to a default).
    """
    if not value:
        raise ValueError("Cannot resolve empty string to ErrorType")

    # 1. Exact value match
    try:
        return ErrorType(value)
    except ValueError:
        pass

    # 2. Exact name match
    try:
        return ErrorType[value]
    except KeyError:
        pass

    # 3. Case-insensitive value match
    lower = value.lower()
    for et in ErrorType:
        if et.value == lower:
            return et

    raise ValueError(
        f"Cannot resolve '{value}' to ErrorType. "
        f"Valid values: {[e.value for e in ErrorType]}, "
        f"valid names: {[e.name for e in ErrorType]}"
    )


# ============================================================
# Default prompt templates per specialization
# ============================================================

ACTOR_STYLE_PROMPTS = {
    ReasoningStyle.ALGEBRAIC: (
        "You are an algebraic reasoner. When solving problems, prefer to set up "
        "equations with variables (e.g., 'let x ='), use symbolic manipulation, "
        "and solve systems of equations when possible."
    ),
    ReasoningStyle.DIRECT: (
        "You are a direct computational reasoner. When solving problems, prefer "
        "straightforward step-by-step numerical calculation without symbolic setup. "
        "Compute each step explicitly."
    ),
    ReasoningStyle.BACKTRACKING: (
        "You are a backtracking reasoner. When solving problems, first attempt a "
        "solution, then verify it by substituting back or checking constraints. "
        "If the verification fails, identify where you went wrong and revise."
    ),
}

CRITIC_SPECIALTY_PROMPTS = {
    ErrorType.ARITHMETIC: (
        "You are a critic specializing in arithmetic error detection. "
        "Carefully check all numerical calculations in the response. "
        "Report any arithmetic mistakes, sign errors, or computation errors."
    ),
    ErrorType.LOGIC: (
        "You are a critic specializing in logical error detection. "
        "Check the reasoning chain for flawed logic, wrong formulas, "
        "incorrect theorem application, or logical fallacies."
    ),
    ErrorType.HALLUCINATION: (
        "You are a critic specializing in hallucination detection. "
        "Check for fabricated numbers, invented theorems, unsupported claims, "
        "or facts that don't match the problem statement."
    ),
    ErrorType.VERIFICATION: (
        "You are a critic specializing in verification failures. "
        "Check if the solution performed adequate self-checks. "
        "Identify cases where the solver should have caught their own error "
        "but didn't verify properly."
    ),
}

# Confidence prompt suffix for Critic (used by MoE Router + weighted voting)
CRITIC_CONFIDENCE_SUFFIX = (
    "\n\nAfter your analysis:\n"
    "1. Judge whether the solver's final answer is correct: "
    "[Answer_Correct: yes] or [Answer_Correct: no]\n"
    "2. Output your confidence in your feedback on a scale of 0.0 to 1.0: "
    "[Confidence: 0.X]"
)


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
    error_specialty: Optional[ErrorType] = None  # For Critics

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
            base = CRITIC_SPECIALTY_PROMPTS.get(self.error_specialty, "")
            return base + CRITIC_CONFIDENCE_SUFFIX
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
            d["error_specialty"] = ErrorType(d["error_specialty"])
        return cls(**d)


# ============================================================
# Agent Registry
# ============================================================

class AgentRegistry:
    """
    Registry for managing the Actor-Critic Society's agents.

    Tracks 3 Actors (ALGEBRAIC, DIRECT, BACKTRACKING) and
    4 Critics (ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION),
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

    def get_critic_by_specialty(self, specialty: ErrorType) -> Optional[AgentConfig]:
        """Get the Critic with a specific error specialty."""
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
        base_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: str = "cache/society",
    ) -> "AgentRegistry":
        """Create a registry with the default 3-Actor + 4-Critic setup."""
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

        # 4 Critics with distinct error specialties
        for specialty in ErrorType:
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
