# Phase 0 Implementation Plan: Diverse Actor-Critic Society Infrastructure

**Status**: In Progress
**Last Updated**: 2025-04-16
**Author**: Planner Agent

---

## Overview

Phase 0 implements the foundational infrastructure for the Diverse Actor-Critic Society experiment, combining ACC-Collab (Actor-Critic collaboration) with Multiagent FT (diverse thinking chains). This phase is divided into two sub-phases:

- **Phase 0.1**: Add Qwen2.5 model support + MATH/GSM8K dataset support
- **Phase 0.2**: Create the Society module (multi-agent deliberation framework)

---

## Phase 0.1: Qwen2.5 + MATH/GSM Support

### 0.1.1 Model Support: Qwen2.5

#### File: `src/training/lora_config.py`

**Current State** (Line 24-29):
```python
MODEL_TARGET_MODULES = {
    "llama3": DEFAULT_TARGET_MODULES,
    "mistral": DEFAULT_TARGET_MODULES,
    "gemma2": DEFAULT_TARGET_MODULES,
    "qwen3": DEFAULT_TARGET_MODULES,
}
```

**Modification**:
```python
MODEL_TARGET_MODULES = {
    "llama3": DEFAULT_TARGET_MODULES,
    "mistral": DEFAULT_TARGET_MODULES,
    "gemma2": DEFAULT_TARGET_MODULES,
    "qwen3": DEFAULT_TARGET_MODULES,
    "qwen2.5": DEFAULT_TARGET_MODULES,  # ADD THIS LINE
}
```

**Notes**:
- Qwen2.5 uses the same attention/FFN structure as other decoder-only models
- Target modules remain identical: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

---

#### File: `src/utils/model_utils.py`

**Current State** (Line 12-49):
```python
def detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    if "llama" in name:
        return "llama3"
    elif "mistral" in name:
        return "mistral"
    elif "gemma" in name:
        return "gemma2"
    elif "qwen3" in name or "qwen" in name:
        return "qwen3"
    else:
        logger.warning(...)
        return "llama3"
```

**Modification**:
```python
def detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    if "llama" in name:
        return "llama3"
    elif "mistral" in name:
        return "mistral"
    elif "gemma" in name:
        return "gemma2"
    elif "qwen2.5" in name or "qwen2_5" in name or "qwen-2.5" in name:
        return "qwen2.5"  # ADD THIS BRANCH (must check before generic "qwen")
    elif "qwen3" in name or "qwen" in name:
        return "qwen3"
    else:
        logger.warning(...)
        return "llama3"
```

**Critical**: The Qwen2.5 check must come BEFORE the generic "qwen" check to avoid misclassification.

---

### 0.1.2 Dataset Support: MATH and GSM8K

#### File: `src/data/loader.py`

**Current State** (Line 14-43):
```python
DATASET_REGISTRY = {
    "boolq": {...},
    "mmlu": {...},
    "bbh": {...},
    "sciq": {...},
    "arc": {...},
}
```

**Modification** (Insert after line 42, before closing brace):
```python
    "math": {
        "hf_id": "competition_math",
        "task_type": "math",
        "splits": ["train", "test"],
    },
    "gsm8k": {
        "hf_id": "gsm8k",
        "task_type": "math",
        "splits": ["train", "test"],
    },
```

**Notes**:
- Both use `task_type: "math"` (new type)
- MATH dataset: `competition_math` on HuggingFace
- GSM8K dataset: `gsm8k` on HuggingFace

---

#### File: `src/data/preprocessor.py`

**Current State** (Line 37-71): Handles `yes_no` and `multiple_choice` task types.

**Modification** (Add new branch after line 44):
```python
    elif task_type == "math":
        # MATH/GSM8K format: question + answer (with \boxed{} for MATH)
        result["question"] = sample.get("question", "")
        result["passage"] = ""  # No passage for math problems
        # Extract answer from various formats
        raw_answer = sample.get("answer", "")
        if isinstance(raw_answer, dict):
            # Some datasets have structured answers
            result["answer"] = raw_answer.get("final", str(raw_answer))
        else:
            result["answer"] = str(raw_answer).strip()
        result["choices"] = []  # No multiple choice
```

**Notes**:
- Math problems don't have passages or multiple choices
- Must handle both string answers and structured dict answers
- Preserve the original answer format for later \boxed{} extraction

---

#### File: `src/prompts/templates.py`

**Current State** (Line 196-204):
```python
DATASET_TEMPLATES = {
    "boolq": BOOLQ_TEMPLATES,
    "mmlu": MMLU_TEMPLATES,
    "bbh": BBH_TEMPLATES,
    "sciq": SCIQ_TEMPLATES,
    "arc": ARC_TEMPLATES,
}
```

**Modification 1**: Add MATH_TEMPLATES (Insert before line 196):
```python
# =============================================================================
# MATH Templates (Math problem solving with \boxed{} answers)
# =============================================================================

MATH_TEMPLATES = {
    PromptType.SINGLE_SHOT: (
        "You will be given a mathematics problem. "
        "You should solve the problem step-by-step, showing your work clearly. "
        "You must provide your final answer within \\boxed{{}} format."
        "\nProblem: {question}"
    ),
    PromptType.GUIDED_SINGLE_SHOT: (
        "You will be given a mathematics problem. "
        "You should solve the problem step-by-step, showing your work clearly, "
        "and arrive at the answer: {target_answer}. "
        "You must provide your final answer within \\boxed{{}} format as \\boxed{{{target_answer}}}."
        "\nProblem: {question}"
    ),
    PromptType.DELIBERATION_ACTOR: (
        "Several people have provided solutions to a mathematics problem. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these solutions into consideration when solving "
        "the following problem. "
        "You should provide a step-by-step solution and give your final answer "
        "within \\boxed{{}} format."
        "\nProblem: {question}"
    ),
    PromptType.GUIDED_DELIBERATION_ACTOR: (
        "Several people have provided solutions to a mathematics problem. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these solutions into consideration when solving "
        "the following problem to arrive at the answer: {target_answer}. "
        "You should provide a step-by-step solution and give your final answer "
        "within \\boxed{{}} format as \\boxed{{{target_answer}}}."
        "\nProblem: {question}"
    ),
    PromptType.DELIBERATION_CRITIC: (
        "I am solving a mathematics problem. "
        "I would like you to help me improve my solution by briefly providing "
        "additional insights or corrections my work may have missed."
        "\nProblem: {question}"
        "\nMy Solution: {actor_response}"
    ),
    PromptType.GUIDED_DELIBERATION_CRITIC: (
        "I would like you to be a deliberation assistant. "
        "You will be given a mathematics problem and my solution. "
        "You should use the problem and my solution to develop additional insights "
        "for why the correct answer is {target_answer}. "
        "Your insights must be brief and must support the fact that the "
        "correct answer is {target_answer}."
        "\nProblem: {question}"
        "\nMy Solution: {actor_response}"
    ),
}

# GSM8K uses the same structure (grade school math, numeric answers)
GSM_TEMPLATES = MATH_TEMPLATES
```

**Modification 2**: Update DATASET_TEMPLATES (after line 204):
```python
DATASET_TEMPLATES = {
    "boolq": BOOLQ_TEMPLATES,
    "mmlu": MMLU_TEMPLATES,
    "bbh": BBH_TEMPLATES,
    "sciq": SCIQ_TEMPLATES,
    "arc": ARC_TEMPLATES,
    "math": MATH_TEMPLATES,      # ADD
    "gsm8k": GSM_TEMPLATES,      # ADD
}
```

---

#### File: `src/algorithms/reward.py`

**Current State** (Line 28-54): Handles `yes_no`, `multiple_choice`, `mixed` task types.

**Modification 1**: Add math task type to extract_answer function:
```python
def extract_answer(response: str, task_type: str = "yes_no") -> Optional[str]:
    if not response or not response.strip():
        return None

    if task_type == "yes_no":
        return _extract_yes_no(response)
    elif task_type == "multiple_choice":
        return _extract_mc(response)
    elif task_type == "math":  # ADD THIS BRANCH
        return _extract_math(response)
    elif task_type == "mixed":
        result = _extract_mc(response)
        if result:
            return result
        return _extract_yes_no(response)
    else:
        logger.warning(f"Unknown task_type: {task_type}")
        return None
```

**Modification 2**: Add _extract_math helper function (Insert after _extract_mc, around line 93):
```python
def _extract_math(text: str) -> Optional[str]:
    """Extract answer from math problem response (handles \boxed{} and numeric answers)."
    # First try: \boxed{answer} format (MATH dataset standard)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    m = re.search(boxed_pattern, text)
    if m:
        return m.group(1).strip()

    # Second try: explicit "final answer is" patterns
    patterns = [
        r'[Ff]inal [Aa]nswer:?\s*([\d\.\,\-\+\s]+)',
        r'[Aa]nswer:?\s*([\d\.\,\-\+\s]+)',
        r'[Tt]he answer is\s*([\d\.\,\-\+\s]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    # Third try: last number in the response (GSM8K numeric answers)
    numbers = re.findall(r'[\d\.\,\-]+', text)
    if numbers:
        return numbers[-1].strip()

    return None
```

**Notes**:
- MATH dataset uses \boxed{} LaTeX format
- GSM8K uses plain numeric answers
- Fallback to last number for robustness

---

### 0.1.3 Configuration Files

#### New File: `configs/data/math.yaml`
```yaml
data:
  name: "math"
  hf_id: "competition_math"
  task_type: "math"
  answer_key: "answer"
```

#### New File: `configs/data/gsm.yaml`
```yaml
data:
  name: "gsm8k"
  hf_id: "gsm8k"
  task_type: "math"
  answer_key: "answer"
```

#### New File: `configs/model/qwen2.5_7b.yaml`
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  type: "qwen2.5"
  max_new_tokens: 512  # Math problems need longer output
  temperature: 0.7
```

---

## Phase 0.2: Society Module Creation

### Architecture Overview

The Society module implements a multi-agent deliberation framework with:
- **Agent Registry**: Manage diverse Actor/Critic personalities
- **Router**: Select agents based on problem characteristics
- **Multi-Deliberation**: Parallel deliberation across agent pairs
- **Data Classifier**: Categorize problems for agent selection
- **Diversity Split**: Partition data across different thinking styles
- **Society Trainer**: Alternating training across multiple agent pairs
- **Inference Pipeline**: End-to-end multi-agent inference

---

### 0.2.1 Core Module: `src/society/__init__.py`

```python
"""
Diverse Actor-Critic Society module.

Implements multi-agent deliberation with diverse thinking chains,
combining ACC-Collab (Actor-Critic collaboration) with Multiagent FT
(diverse thinking chains for complex reasoning).

Key concepts:
- Agent Registry: Manage diverse Actor/Critic personalities
- Router: Select agents based on problem characteristics
- Multi-Deliberation: Parallel deliberation across agent pairs
- Data Classifier: Categorize problems for agent selection
"""

from __future__ import annotations

from src.society.agent_registry import (
    AgentRegistry,
    AgentType,
    ThinkingStyle,
)
from src.society.router import Router, RoutingStrategy
from src.society.multi_deliberation import multi_agent_deliberate
from src.society.data_classifier import DataClassifier, ProblemCategory
from src.society.diversity_split import DiversitySplit

__all__ = [
    "AgentRegistry",
    "AgentType",
    "ThinkingStyle",
    "Router",
    "RoutingStrategy",
    "multi_agent_deliberate",
    "DataClassifier",
    "ProblemCategory",
    "DiversitySplit",
]
```

---

### 0.2.2 Agent Registry: `src/society/agent_registry.py`

```python
"""
Agent Registry for managing diverse Actor/Critic personalities.

Defines agent types and thinking styles following Multiagent FT paper:
- Analytical: Step-by-step logical reasoning
- Intuitive: Quick pattern recognition
- Creative: Lateral thinking and analogies
- Skeptical: Devil's advocate, verification focus
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Type of agent in the Actor-Critic framework."""
    ACTOR = "actor"
    CRITIC = "critic"


class ThinkingStyle(Enum):
    """Thinking styles for diverse agents (from Multiagent FT)."""
    ANALYTICAL = "analytical"      # Step-by-step decomposition
    INTUITIVE = "intuitive"        # Pattern recognition
    CREATIVE = "creative"          # Lateral thinking
    SKEPTICAL = "skeptical"        # Verification focus
    COLLABORATIVE = "collaborative"  # Building on others' ideas
    PRECISE = "precise"            # Detail-oriented


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    agent_type: AgentType
    thinking_style: ThinkingStyle
    model_path: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 256

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on thinking style."""
        style_prompts = {
            ThinkingStyle.ANALYTICAL: (
                "You are an analytical thinker. Break down problems step-by-step, "
                "identify key components, and use logical reasoning to derive solutions."
            ),
            ThinkingStyle.INTUITIVE: (
                "You are an intuitive thinker. Recognize patterns quickly and use "
                "your judgment to arrive at solutions efficiently."
            ),
            ThinkingStyle.CREATIVE: (
                "You are a creative thinker. Use lateral thinking, analogies, and "
                "novel perspectives to approach problems from unique angles."
            ),
            ThinkingStyle.SKEPTICAL: (
                "You are a skeptical thinker. Question assumptions, verify claims, "
                "and identify potential flaws in reasoning."
            ),
            ThinkingStyle.COLLABORATIVE: (
                "You are a collaborative thinker. Build upon others' ideas and "
                "synthesize different perspectives into coherent solutions."
            ),
            ThinkingStyle.PRECISE: (
                "You are a precise thinker. Pay attention to details, ensure accuracy, "
                "and provide exact, well-structured solutions."
            ),
        }
        return style_prompts.get(
            self.thinking_style,
            "You are a helpful assistant."
        )


class AgentRegistry:
    """
    Registry for managing diverse Actor and Critic agents.

    Provides methods to register, retrieve, and query agents based on
    their type, thinking style, and other attributes.
    """

    def __init__(self):
        self._actors: dict[str, AgentConfig] = {}
        self._critics: dict[str, AgentConfig] = {}

    def register(self, config: AgentConfig) -> None:
        """Register an agent configuration."""
        if config.agent_type == AgentType.ACTOR:
            self._actors[config.name] = config
        else:
            self._critics[config.name] = config
        logger.info(
            f"Registered {config.agent_type.value}: {config.name} "
            f"({config.thinking_style.value})"
        )

    def get_actor(self, name: str) -> Optional[AgentConfig]:
        """Get an actor by name."""
        return self._actors.get(name)

    def get_critic(self, name: str) -> Optional[AgentConfig]:
        """Get a critic by name."""
        return self._critics.get(name)

    def list_actors(self, style: Optional[ThinkingStyle] = None) -> list[AgentConfig]:
        """List all actors, optionally filtered by thinking style."""
        actors = list(self._actors.values())
        if style:
            actors = [a for a in actors if a.thinking_style == style]
        return actors

    def list_critics(self, style: Optional[ThinkingStyle] = None) -> list[AgentConfig]:
        """List all critics, optionally filtered by thinking style."""
        critics = list(self._critics.values())
        if style:
            critics = [c for c in critics if c.thinking_style == style]
        return critics

    def get_pairs(self) -> list[tuple[AgentConfig, AgentConfig]]:
        """Get all valid actor-critic pairs."""
        pairs = []
        for actor in self._actors.values():
            for critic in self._critics.values():
                pairs.append((actor, critic))
        return pairs

    @classmethod
    def from_config(cls, config_path: str) -> "AgentRegistry":
        """Load agent registry from YAML config file."""
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        registry = cls()
        for name, agent_cfg in cfg.get("actors", {}).items():
            config = AgentConfig(
                name=name,
                agent_type=AgentType.ACTOR,
                thinking_style=ThinkingStyle(agent_cfg["thinking_style"]),
                model_path=agent_cfg["model_path"],
                system_prompt=agent_cfg.get("system_prompt", ""),
                temperature=agent_cfg.get("temperature", 0.7),
                max_tokens=agent_cfg.get("max_tokens", 256),
            )
            registry.register(config)

        for name, agent_cfg in cfg.get("critics", {}).items():
            config = AgentConfig(
                name=name,
                agent_type=AgentType.CRITIC,
                thinking_style=ThinkingStyle(agent_cfg["thinking_style"]),
                model_path=agent_cfg["model_path"],
                system_prompt=agent_cfg.get("system_prompt", ""),
                temperature=agent_cfg.get("temperature", 0.7),
                max_tokens=agent_cfg.get("max_tokens", 256),
            )
            registry.register(config)

        return registry
```

---

### 0.2.3 Router: `src/society/router.py`

```python
"""
Router for selecting appropriate agents based on problem characteristics.

Implements different routing strategies:
- RANDOM: Uniform random selection
- ROUND_ROBIN: Cyclic selection
- CLASSIFIER_BASED: Use trained classifier to predict best agent
- ENSEMBLE: Combine multiple agents' outputs
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.society.agent_registry import AgentConfig, AgentRegistry, ThinkingStyle

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Strategy for routing problems to agents."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    CLASSIFIER_BASED = "classifier_based"
    ENSEMBLE = "ensemble"


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    actor_configs: list[AgentConfig]
    critic_configs: list[AgentConfig]
    strategy: RoutingStrategy
    metadata: dict[str, Any] = field(default_factory=dict)


class Router:
    """
    Router for selecting agents based on problem characteristics.

    Can be configured with different routing strategies and can
    optionally use a trained classifier for intelligent routing.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        strategy: RoutingStrategy = RoutingStrategy.RANDOM,
        classifier: Optional[Any] = None,
    ):
        self.registry = registry
        self.strategy = strategy
        self.classifier = classifier
        self._round_robin_idx = 0

    def route(
        self,
        sample: dict,
        num_actors: int = 1,
        num_critics: int = 1,
    ) -> RoutingDecision:
        """
        Route a problem to appropriate agents.

        Args:
            sample: Standardized sample dict (question, passage, etc.).
            num_actors: Number of actors to select.
            num_critics: Number of critics to select.

        Returns:
            RoutingDecision with selected agent configs.
        """
        if self.strategy == RoutingStrategy.RANDOM:
            return self._route_random(num_actors, num_critics)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(num_actors, num_critics)
        elif self.strategy == RoutingStrategy.CLASSIFIER_BASED:
            if self.classifier is None:
                logger.warning("Classifier not provided, falling back to random")
                return self._route_random(num_actors, num_critics)
            return self._route_classifier(sample, num_actors, num_critics)
        elif self.strategy == RoutingStrategy.ENSEMBLE:
            return self._route_ensemble(num_actors, num_critics)
        else:
            logger.error(f"Unknown strategy: {self.strategy}")
            return self._route_random(num_actors, num_critics)

    def _route_random(
        self, num_actors: int, num_critics: int
    ) -> RoutingDecision:
        """Random routing strategy."""
        actors = self.registry.list_actors()
        critics = self.registry.list_critics()

        selected_actors = random.sample(actors, min(num_actors, len(actors)))
        selected_critics = random.sample(critics, min(num_critics, len(critics)))

        return RoutingDecision(
            actor_configs=selected_actors,
            critic_configs=selected_critics,
            strategy=RoutingStrategy.RANDOM,
        )

    def _route_round_robin(
        self, num_actors: int, num_critics: int
    ) -> RoutingDecision:
        """Round-robin routing strategy."""
        actors = self.registry.list_actors()
        critics = self.registry.list_critics()

        n = len(actors)
        selected_actors = [
            actors[(self._round_robin_idx + i) % n] for i in range(num_actors)
        ]
        self._round_robin_idx = (self._round_robin_idx + num_actors) % n

        m = len(critics)
        selected_critics = [
            critics[(self._round_robin_idx + i) % m] for i in range(num_critics)
        ]

        return RoutingDecision(
            actor_configs=selected_actors,
            critic_configs=selected_critics,
            strategy=RoutingStrategy.ROUND_ROBIN,
            metadata={"index": self._round_robin_idx},
        )

    def _route_classifier(
        self, sample: dict, num_actors: int, num_critics: int
    ) -> RoutingDecision:
        """Classifier-based routing strategy."""
        # TODO: Implement classifier-based routing
        # For now, fall back to random
        return self._route_random(num_actors, num_critics)

    def _route_ensemble(
        self, num_actors: int, num_critics: int
    ) -> RoutingDecision:
        """Ensemble routing: select all available agents."""
        return RoutingDecision(
            actor_configs=self.registry.list_actors(),
            critic_configs=self.registry.list_critics(),
            strategy=RoutingStrategy.ENSEMBLE,
        )
```

---

### 0.2.4 Multi-Deliberation: `src/society/multi_deliberation.py`

```python
"""
Multi-agent deliberation engine.

Runs parallel deliberation across multiple Actor-Critic pairs,
then aggregates their responses using voting or other strategies.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from src.algorithms.deliberation import deliberate, guided_deliberate_round
from src.algorithms.reward import extract_answer
from src.society.agent_registry import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class DeliberationResult:
    """Result from a single Actor-Critic pair deliberation."""
    actor_name: str
    critic_name: str
    trajectory: list[dict]
    final_answer: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated result from multiple Actor-Critic pairs."""
    individual_results: list[DeliberationResult]
    final_answer: str
    confidence: float
    aggregation_method: str
    metadata: dict[str, Any] = field(default_factory=dict)


def multi_agent_deliberate(
    models: dict[str, Any],  # name -> VLLMInference
    actor_configs: list[AgentConfig],
    critic_configs: list[AgentConfig],
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    aggregation: str = "majority_vote",
    parallel: bool = True,
) -> AggregatedResult:
    """
    Run deliberation across multiple Actor-Critic pairs in parallel.

    Args:
        models: Dict mapping agent names to VLLMInference instances.
        actor_configs: List of Actor configurations.
        critic_configs: List of Critic configurations.
        sample: Standardized sample dict.
        dataset_name: Dataset name for prompt selection.
        num_rounds: Number of deliberation rounds per pair.
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        aggregation: How to aggregate answers ("majority_vote", "weighted", "best").
        parallel: Whether to run pairs in parallel.

    Returns:
        AggregatedResult with final answer and confidence.
    """
    # Generate all Actor-Critic pairs
    pairs = [(a, c) for a in actor_configs for c in critic_configs]

    results = []

    if parallel:
        # Run pairs in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(pairs), 8)) as executor:
            futures = {
                executor.submit(
                    _run_single_pair,
                    models,
                    actor,
                    critic,
                    sample,
                    dataset_name,
                    num_rounds,
                    max_tokens,
                    temperature,
                ): (actor, critic)
                for actor, critic in pairs
            }

            for future in as_completed(futures):
                actor, critic = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Pair {actor.name}-{critic.name} failed: {e}")
    else:
        # Run sequentially
        for actor, critic in pairs:
            result = _run_single_pair(
                models,
                actor,
                critic,
                sample,
                dataset_name,
                num_rounds,
                max_tokens,
                temperature,
            )
            results.append(result)

    # Aggregate results
    return _aggregate_results(results, aggregation)


def _run_single_pair(
    models: dict[str, Any],
    actor: AgentConfig,
    critic: AgentConfig,
    sample: dict,
    dataset_name: str,
    num_rounds: int,
    max_tokens: int,
    temperature: float,
) -> DeliberationResult:
    """Run deliberation for a single Actor-Critic pair."""
    actor_model = models.get(actor.name)
    critic_model = models.get(critic.name)

    if actor_model is None or critic_model is None:
        raise ValueError(f"Model not found for {actor.name} or {critic.name}")

    # Run standard deliberation
    trajectory = deliberate(
        actor_model,
        critic_model,
        sample,
        dataset_name,
        num_rounds=num_rounds,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Extract final answer from last round
    final_round = trajectory[-1] if trajectory else {}
    final_answer = final_round.get("actor_answer")

    return DeliberationResult(
        actor_name=actor.name,
        critic_name=critic.name,
        trajectory=trajectory,
        final_answer=final_answer,
        metadata={
            "actor_style": actor.thinking_style.value,
            "critic_style": critic.thinking_style.value,
        },
    )


def _aggregate_results(
    results: list[DeliberationResult],
    method: str = "majority_vote",
) -> AggregatedResult:
    """Aggregate results from multiple Actor-Critic pairs."""
    if not results:
        raise ValueError("No results to aggregate")

    if method == "majority_vote":
        return _majority_vote(results)
    elif method == "weighted":
        return _weighted_aggregation(results)
    elif method == "best":
        return _select_best(results)
    else:
        logger.warning(f"Unknown aggregation method: {method}, using majority_vote")
        return _majority_vote(results)


def _majority_vote(results: list[DeliberationResult]) -> AggregatedResult:
    """Aggregate using majority voting."""
    from collections import Counter

    answers = [r.final_answer for r in results if r.final_answer]
    if not answers:
        return AggregatedResult(
            individual_results=results,
            final_answer="",
            confidence=0.0,
            aggregation_method="majority_vote",
        )

    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    final_answer = most_common[0]
    confidence = most_common[1] / len(answers)

    return AggregatedResult(
        individual_results=results,
        final_answer=final_answer,
        confidence=confidence,
        aggregation_method="majority_vote",
    )


def _weighted_aggregation(results: list[DeliberationResult]) -> AggregatedResult:
    """Aggregate using weighted voting (placeholder for future implementation)."""
    # For now, same as majority vote
    return _majority_vote(results)


def _select_best(results: list[DeliberationResult]) -> AggregatedResult:
    """Select the best result based on some heuristic (placeholder)."""
    # For now, just return the first result
    return AggregatedResult(
        individual_results=results,
        final_answer=results[0].final_answer or "",
        confidence=1.0,
        aggregation_method="best",
    )
```

---

### 0.2.5 Data Classifier: `src/society/data_classifier.py`

```python
"""
Data classifier for categorizing problems by difficulty and type.

Used to route problems to appropriate agents with specialized thinking styles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ProblemCategory(Enum):
    """Categories of problems for routing."""
    SIMPLE_FACTUAL = "simple_factual"      # Direct recall, single step
    COMPLEX_REASONING = "complex_reasoning"  # Multi-step logical reasoning
    MATHEMATICAL = "mathematical"          # Quantitative reasoning
    CREATIVE = "creative"                  # Open-ended, lateral thinking
    VERIFICATION = "verification"          # Fact-checking, skepticism needed


@dataclass
class ClassificationResult:
    """Result of classifying a problem."""
    category: ProblemCategory
    difficulty: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataClassifier:
    """
    Classifier for categorizing problems based on their characteristics.

    Uses heuristic rules for now, but can be extended with trained models.
    """

    def classify(self, sample: dict, dataset_name: str) -> ClassificationResult:
        """
        Classify a problem sample.

        Args:
            sample: Standardized sample dict.
            dataset_name: Name of the dataset.

        Returns:
            ClassificationResult with category and difficulty.
        """
        # Dataset-based heuristics
        if dataset_name in ("math", "gsm8k"):
            return ClassificationResult(
                category=ProblemCategory.MATHEMATICAL,
                difficulty=0.7,
                confidence=0.9,
                metadata={"reason": "dataset_type"},
            )

        question = sample.get("question", "").lower()

        # Length-based difficulty heuristic
        question_length = len(question.split())
        difficulty = min(1.0, question_length / 50.0)

        # Keyword-based classification
        if any(w in question for w in ["calculate", "solve", "equation", "number"]):
            category = ProblemCategory.MATHEMATICAL
        elif any(w in question for w in ["verify", "true", "false", "correct"]):
            category = ProblemCategory.VERIFICATION
        elif question_length < 15:
            category = ProblemCategory.SIMPLE_FACTUAL
        elif question_length > 40:
            category = ProblemCategory.COMPLEX_REASONING
        else:
            category = ProblemCategory.COMPLEX_REASONING

        return ClassificationResult(
            category=category,
            difficulty=difficulty,
            confidence=0.7,
            metadata={
                "question_length": question_length,
                "reason": "heuristic",
            },
        )
```

---

### 0.2.6 Diversity Split: `src/society/diversity_split.py`

```python
"""
Diversity split: Partition data across different thinking styles.

Implements the diversification strategy from Multiagent FT:
split data by the type of reasoning required and assign to
specialized Actor-Critic pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.society.data_classifier import DataClassifier, ProblemCategory

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Result of splitting data across thinking styles."""
    splits: dict[str, list[dict]]  # thinking_style -> samples
    metadata: dict


class DiversitySplit:
    """
    Split dataset across different thinking styles based on problem characteristics.

    Each Actor-Critic pair specializes in a particular type of reasoning.
    """

    def __init__(
        self,
        classifier: Optional[DataClassifier] = None,
        balance: bool = True,
        seed: int = 42,
    ):
        self.classifier = classifier or DataClassifier()
        self.balance = balance
        self.rng = np.random.default_rng(seed)

    def split(
        self,
        samples: list[dict],
        dataset_name: str,
        thinking_styles: list[str],
    ) -> SplitResult:
        """
        Split samples across thinking styles.

        Args:
            samples: List of standardized sample dicts.
            dataset_name: Name of the dataset.
            thinking_styles: List of thinking style names to split across.

        Returns:
            SplitResult with samples assigned to each style.
        """
        # Classify all samples
        classifications = []
        for sample in samples:
            result = self.classifier.classify(sample, dataset_name)
            classifications.append((sample, result))

        # Map categories to thinking styles
        category_to_style = self._map_categories_to_styles(thinking_styles)

        # Assign samples to styles
        splits = {style: [] for style in thinking_styles}

        for sample, classification in classifications:
            style = category_to_style.get(
                classification.category,
                thinking_styles[0],  # Default to first style
            )
            splits[style].append(sample)

        # Balance splits if requested
        if self.balance:
            splits = self._balance_splits(splits)

        logger.info(
            f"Data split: {', '.join(f'{k}={len(v)}' for k, v in splits.items())}"
        )

        return SplitResult(
            splits=splits,
            metadata={
                "classifier": str(type(self.classifier).__name__),
                "balanced": self.balance,
            },
        )

    def _map_categories_to_styles(
        self, thinking_styles: list[str]
    ) -> dict[ProblemCategory, str]:
        """Map problem categories to thinking styles."""
        # Simple mapping: can be made configurable
        mapping = {
            ProblemCategory.MATHEMATICAL: "analytical",
            ProblemCategory.COMPLEX_REASONING: "analytical",
            ProblemCategory.SIMPLE_FACTUAL: "intuitive",
            ProblemCategory.VERIFICATION: "skeptical",
            ProblemCategory.CREATIVE: "creative",
        }

        # Ensure all mapped styles exist in thinking_styles
        valid_mapping = {}
        for cat, style in mapping.items():
            if style in thinking_styles:
                valid_mapping[cat] = style
            elif thinking_styles:
                valid_mapping[cat] = thinking_styles[0]

        return valid_mapping

    def _balance_splits(
        self, splits: dict[str, list[dict]]
    ) -> dict[str, list[dict]]:
        """Balance splits to have approximately equal size."""
        if not splits:
            return splits

        target_size = min(len(v) for v in splits.values())

        balanced = {}
        for style, samples in splits.items():
            if len(samples) > target_size:
                # Downsample
                indices = self.rng.choice(
                    len(samples), size=target_size, replace=False
                )
                balanced[style] = [samples[i] for i in indices]
            else:
                balanced[style] = samples

        return balanced
```

---

### 0.2.7 Society Trainer: `src/society/society_trainer.py`

```python
"""
Society trainer: Alternating training across multiple Actor-Critic pairs.

Extends the ACC-Collab alternating training to support multiple
specialized agent pairs with different thinking styles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.society.agent_registry import AgentConfig, AgentRegistry
from src.society.diversity_split import DiversitySplit

logger = logging.getLogger(__name__)


@dataclass
class SocietyTrainingResult:
    """Result of society training."""
    actor_paths: dict[str, str]  # actor_name -> checkpoint path
    critic_paths: dict[str, str]  # critic_name -> checkpoint path
    metrics: dict[str, Any]


class SocietyTrainer:
    """
    Train multiple Actor-Critic pairs in an alternating fashion.

    Each iteration:
    1. Split data by thinking style (DiversitySplit)
    2. Train each Critic on its assigned data
    3. Train each Actor on its assigned data
    """

    def __init__(
        self,
        registry: AgentRegistry,
        output_base_dir: str,
        diversity_split: Optional[DiversitySplit] = None,
    ):
        self.registry = registry
        self.output_base_dir = output_base_dir
        self.diversity_split = diversity_split or DiversitySplit()

    def train(
        self,
        dataset: list[dict],
        dataset_name: str,
        num_iterations: int = 1,
        num_rounds: int = 5,
        num_simulations: int = 5,
        reward_threshold: float = 0.0,
        lora_r: int = 256,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        num_epochs: int = 1,
        max_length: int = 2048,
        beta: float = 0.1,
        seed: int = 42,
        val_dataset: Optional[list[dict]] = None,
    ) -> SocietyTrainingResult:
        """
        Train all Actor-Critic pairs in the society.

        Args:
            dataset: Training dataset.
            dataset_name: Name of the dataset.
            num_iterations: Number of alternating training iterations.
            num_rounds: Number of deliberation rounds.
            num_simulations: Monte Carlo roll-out simulations.
            reward_threshold: Threshold for preference pair filtering.
            lora_r: LoRA rank.
            learning_rate: Learning rate.
            batch_size: Batch size.
            num_epochs: Number of epochs.
            max_length: Max sequence length.
            beta: DPO beta parameter.
            seed: Random seed.
            val_dataset: Optional validation dataset.

        Returns:
            SocietyTrainingResult with checkpoint paths and metrics.
        """
        # Get all thinking styles
        thinking_styles = list(set(
            a.thinking_style.value for a in self.registry.list_actors()
        ))

        # Split data across thinking styles
        split_result = self.diversity_split.split(
            dataset, dataset_name, thinking_styles
        )

        # Train each Actor-Critic pair
        actor_paths = {}
        critic_paths = {}
        metrics = {}

        for style in thinking_styles:
            style_data = split_result.splits.get(style, [])
            if not style_data:
                logger.warning(f"No data for style: {style}")
                continue

            # Get actors and critics for this style
            actors = [a for a in self.registry.list_actors()
                     if a.thinking_style.value == style]
            critics = [c for c in self.registry.list_critics()
                      if c.thinking_style.value == style]

            if not actors or not critics:
                logger.warning(f"No agents for style: {style}")
                continue

            # For simplicity, train first actor-critic pair per style
            actor = actors[0]
            critic = critics[0]

            logger.info(f"Training {actor.name} + {critic.name} ({style})")

            # Import training functions
            from src.training.scheduler import alternating_train

            result = alternating_train(
                actor_path=actor.model_path,
                critic_path=critic.model_path,
                dataset=style_data,
                dataset_name=dataset_name,
                output_base_dir=f"{self.output_base_dir}/{style}",
                num_iterations=num_iterations,
                num_rounds=num_rounds,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                lora_r=lora_r,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                max_length=max_length,
                beta=beta,
                seed=seed,
                val_dataset=val_dataset,
            )

            actor_paths[actor.name] = result["actor_path"]
            critic_paths[critic.name] = result["critic_path"]
            metrics[style] = result.get("metrics", {})

        return SocietyTrainingResult(
            actor_paths=actor_paths,
            critic_paths=critic_paths,
            metrics=metrics,
        )
```

---

### 0.2.8 Inference Pipeline: `src/society/inference_pipeline.py`

```python
"""
Inference pipeline for the Diverse Actor-Critic Society.

Provides end-to-end inference from problem selection to final answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.inference.vllm_server import VLLMInference
from src.society.agent_registry import AgentRegistry
from src.society.data_classifier import DataClassifier
from src.society.multi_deliberation import multi_agent_deliberate, AggregatedResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of society inference."""
    final_answer: str
    confidence: float
    selected_actors: list[str]
    selected_critics: list[str]
    individual_results: list[dict]
    metadata: dict[str, Any]


class SocietyInferencePipeline:
    """
    End-to-end inference pipeline for multi-agent deliberation.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        classifier: Optional[DataClassifier] = None,
    ):
        self.registry = registry
        self.classifier = classifier or DataClassifier()
        self.models: dict[str, VLLMInference] = {}

    def load_models(
        self,
        dtype: str = "float32",
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
    ) -> None:
        """Load all registered models into memory."""
        # Collect unique model paths
        model_paths = set()
        for agent in self.registry.list_actors():
            model_paths.add(agent.model_path)
        for agent in self.registry.list_critics():
            model_paths.add(agent.model_path)

        # Load each unique model once
        for path in model_paths:
            logger.info(f"Loading model: {path}")
            self.models[path] = VLLMInference(
                path,
                cuda_device=0,  # TODO: Support multiple GPUs
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )

        # Create name -> model mapping
        self.name_to_model = {}
        for agent in self.registry.list_actors():
            self.name_to_model[agent.name] = self.models[agent.model_path]
        for agent in self.registry.list_critics():
            self.name_to_model[agent.name] = self.models[agent.model_path]

    def infer(
        self,
        sample: dict,
        dataset_name: str,
        num_rounds: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.7,
        aggregation: str = "majority_vote",
    ) -> InferenceResult:
        """
        Run inference on a sample.

        Args:
            sample: Standardized sample dict.
            dataset_name: Name of the dataset.
            num_rounds: Number of deliberation rounds.
            max_tokens: Max tokens per generation.
            temperature: Sampling temperature.
            aggregation: Aggregation method for results.

        Returns:
            InferenceResult with final answer and metadata.
        """
        # Classify problem
        classification = self.classifier.classify(sample, dataset_name)

        # Select agents based on classification
        selected_style = self._select_agents_for_category(classification.category)
        actors = [a for a in self.registry.list_actors()
                 if a.thinking_style.value == selected_style]
        critics = [c for c in self.registry.list_critics()
                  if c.thinking_style.value == selected_style]

        # Run multi-agent deliberation
        result = multi_agent_deliberate(
            models=self.name_to_model,
            actor_configs=actors,
            critic_configs=critics,
            sample=sample,
            dataset_name=dataset_name,
            num_rounds=num_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
            aggregation=aggregation,
        )

        return InferenceResult(
            final_answer=result.final_answer,
            confidence=result.confidence,
            selected_actors=[a.name for a in result.individual_results],
            selected_critics=[c.critic_name for c in result.individual_results],
            individual_results=[r.__dict__ for r in result.individual_results],
            metadata={
                "classification": classification.__dict__,
                "aggregation_method": aggregation,
            },
        )

    def _select_agents_for_category(self, category) -> str:
        """Select thinking style based on problem category."""
        from src.society.data_classifier import ProblemCategory

        mapping = {
            ProblemCategory.MATHEMATICAL: "analytical",
            ProblemCategory.COMPLEX_REASONING: "analytical",
            ProblemCategory.SIMPLE_FACTUAL: "intuitive",
            ProblemCategory.VERIFICATION: "skeptical",
            ProblemCategory.CREATIVE: "creative",
        }
        return mapping.get(category, "analytical")

    def cleanup(self) -> None:
        """Clean up loaded models."""
        for model in self.models.values():
            del model
        self.models.clear()
        self.name_to_model.clear()
```

---

### 0.2.9 Configuration Files

#### New File: `configs/society/base.yaml`
```yaml
# Society module base configuration

defaults:
  - _self_

# Society settings
society:
  num_actors_per_style: 1
  num_critics_per_style: 1
  thinking_styles:
    - analytical
    - intuitive
    - creative
    - skeptical

# Routing settings
routing:
  strategy: "random"  # random, round_robin, classifier_based, ensemble
  classifier_path: null

# Training settings
training:
  num_iterations: 2
  num_rounds: 5
  num_simulations: 5
  reward_threshold: 0.0
  lora_r: 256
  lora_alpha: 512
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  max_length: 2048
  beta: 0.1

# Diversity settings
diversity:
  balance_splits: true
  seed: 42

# Inference settings
inference:
  aggregation: "majority_vote"  # majority_vote, weighted, best
  parallel: true
  max_workers: 8
```

#### New File: `configs/society/actors.yaml`
```yaml
# Actor configurations for diverse thinking styles

actors:
  actor_analytical:
    thinking_style: "analytical"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.7
    max_tokens: 256

  actor_intuitive:
    thinking_style: "intuitive"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.8
    max_tokens: 256

  actor_creative:
    thinking_style: "creative"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.9
    max_tokens: 256

  actor_skeptical:
    thinking_style: "skeptical"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.6
    max_tokens: 256
```

#### New File: `configs/society/critics.yaml`
```yaml
# Critic configurations for diverse thinking styles

critics:
  critic_analytical:
    thinking_style: "analytical"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.7
    max_tokens: 256

  critic_intuitive:
    thinking_style: "intuitive"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.8
    max_tokens: 256

  critic_creative:
    thinking_style: "creative"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.9
    max_tokens: 256

  critic_skeptical:
    thinking_style: "skeptical"
    model_path: "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: ""
    temperature: 0.6
    max_tokens: 256
```

#### New File: `configs/society/router.yaml`
```yaml
# Router configuration

router:
  strategy: "classifier_based"  # random, round_robin, classifier_based, ensemble
  classifier:
    type: "heuristic"  # heuristic, trained
    model_path: null
  ensemble:
    voting: "majority"  # majority, weighted
    confidence_threshold: 0.5
```

#### New File: `configs/society/experiment_h100.yaml`
```yaml
# Society experiment configuration for single H100 GPU

common:
  cache_dir: "experiments/society_math"
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  dataset: "math"
  seed: 42
  max_samples: 100
  use_wandb: false

step01_bootstrap:
  output_dir: "experiments/society_math/bootstrap"
  num_rounds: 5
  num_simulations: 5
  reward_threshold: 0.0
  max_tokens: 512
  temperature: 0.7
  device: 0
  dtype: "bfloat16"
  gpu_memory_utilization: 0.85

step02_classify:
  input_dir: "experiments/society_math/bootstrap"
  output_dir: "experiments/society_math/classified"
  classifier_type: "heuristic"

step03_diversify_actors:
  input_dir: "experiments/society_math/classified"
  output_dir: "experiments/society_math/actors"
  thinking_styles: ["analytical", "intuitive", "creative", "skeptical"]
  lora_r: 256
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  max_length: 2048
  device: 0

step04_diversify_critics:
  input_dir: "experiments/society_math/classified"
  output_dir: "experiments/society_math/critics"
  thinking_styles: ["analytical", "intuitive", "creative", "skeptical"]
  lora_r: 256
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  max_length: 2048
  device: 0

step05_train_society:
  actor_base_dir: "experiments/society_math/actors"
  critic_base_dir: "experiments/society_math/critics"
  output_dir: "experiments/society_math/society"
  num_iterations: 2
  num_rounds: 5
  num_simulations: 5
  reward_threshold: 0.0
  lora_r: 256
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  max_length: 2048
  beta: 0.1
  device: 0
  dtype: "bfloat16"
  gpu_memory_utilization: 0.85

step06_evaluate:
  society_dir: "experiments/society_math/society"
  output_dir: "experiments/society_math/eval"
  num_rounds: 5
  max_tokens: 512
  temperature: 0.7
  device: 0
  dtype: "bfloat16"
  gpu_memory_utilization: 0.85
  aggregation: "majority_vote"
```

---

## Implementation Order & Dependencies

### Phase 0.1 Order (can be done in parallel):
1. `src/training/lora_config.py` - Add qwen2.5 to MODEL_TARGET_MODULES
2. `src/utils/model_utils.py` - Add qwen2.5 detection branch
3. `src/data/loader.py` - Add math/gsm8k to DATASET_REGISTRY
4. `src/data/preprocessor.py` - Add math task_type handling
5. `src/prompts/templates.py` - Add MATH_TEMPLATES and GSM_TEMPLATES
6. `src/algorithms/reward.py` - Add _extract_math function
7. `configs/data/math.yaml` - Create
8. `configs/data/gsm.yaml` - Create
9. `configs/model/qwen2.5_7b.yaml` - Create

### Phase 0.2 Order (sequential due to dependencies):
1. `src/society/__init__.py` - Create (no dependencies)
2. `src/society/agent_registry.py` - Create (no dependencies)
3. `src/society/data_classifier.py` - Create (no dependencies)
4. `src/society/router.py` - Create (depends on agent_registry)
5. `src/society/diversity_split.py` - Create (depends on data_classifier)
6. `src/society/multi_deliberation.py` - Create (depends on agent_registry)
7. `src/society/society_trainer.py` - Create (depends on diversity_split)
8. `src/society/inference_pipeline.py` - Create (depends on all above)
9. Configuration files - Create all

---

## Testing Checklist

After completing Phase 0, verify:

### Phase 0.1 Tests:
- [ ] `detect_model_type("Qwen/Qwen2.5-7B-Instruct")` returns `"qwen2.5"`
- [ ] `get_lora_config("qwen2.5")` returns valid config
- [ ] `load_dataset("math")` loads competition_math successfully
- [ ] `load_dataset("gsm8k")` loads gsm8k successfully
- [ ] `standardize_sample(sample, "math")` handles math format
- [ ] `get_prompt_template("math", SINGLE_SHOT)` returns valid template
- [ ] `extract_answer("The answer is \\boxed{42}", "math")` returns `"42"`

### Phase 0.2 Tests:
- [ ] `AgentRegistry.from_config()` loads agents from YAML
- [ ] `Router.route()` returns valid RoutingDecision
- [ ] `DataClassifier.classify()` returns ProblemCategory
- [ ] `DiversitySplit.split()` partitions data correctly
- [ ] `SocietyInferencePipeline.load_models()` loads all agents
- [ ] Configuration files load without errors

---

## Notes & Considerations

1. **Qwen vs Qwen2.5**: The existing code has "qwen3" detection. Qwen2.5 is a distinct model family. Ensure the detection order doesn't cause misclassification.

2. **Math Answer Formats**: MATH uses `\boxed{}` LaTeX, GSM8K uses plain numbers. The `_extract_math` function handles both.

3. **Society Module Memory**: Loading multiple models simultaneously is memory-intensive. The `SocietyInferencePipeline` should support model sharing (same base model for multiple agents).

4. **Parallelism**: Multi-agent deliberation uses `ThreadPoolExecutor`. For GPU-heavy workloads, consider using `ProcessPoolExecutor` or async approaches.

5. **Single GPU Constraint**: Since only 1 H100 is available, the Society module should be designed to:
   - Share base models across agents with different LoRA adapters
   - Use sequential inference when parallel isn't feasible
   - Support gradient checkpointing for memory efficiency

---

**Document Version**: 1.0
**Next Review**: After Phase 0.1 completion
