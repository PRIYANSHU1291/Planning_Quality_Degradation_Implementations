"""
Abstract base class for all planning agents.

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the planning quality study.

    Provides rule-based action generation (no LLM required) with configurable
    degradation profiles that simulate realistic agent behaviour over long horizons.

    Args:
        name: Human-readable agent name.
        seed: Random seed for reproducibility.
    """

    def __init__(self, name: str, seed: int = 42) -> None:
        self.name = name
        self.seed = seed
        self._rng = random.Random(seed)

        # Trajectory tracking
        self.action_history: List[Dict[str, Any]] = []
        self.reasoning_history: List[str] = []
        self.observation_history: List[Dict[str, Any]] = []
        self.error_count: int = 0
        self.context_buffer: List[str] = []
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce an action given the current observation.

        Args:
            observation: Dict from MSRPBEnvironment with 'text', 'task_type',
                         'valid_actions', 'goal', 'context_size', etc.

        Returns:
            Action dict with 'type', 'reasoning', 'target'.
        """

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state for a new episode."""
        self._rng = random.Random(self.seed)
        self.action_history = []
        self.reasoning_history = []
        self.observation_history = []
        self.error_count = 0
        self.context_buffer = []
        self.step_count = 0

    def observe(self, observation: Dict[str, Any]) -> None:
        """Record an observation (called after environment.step)."""
        self.observation_history.append(observation)
        obs_text = observation.get("text", "")
        if obs_text:
            self.context_buffer.append(obs_text)

    def record_action(self, action: Dict[str, Any], valid: bool = True) -> None:
        """Record an action taken."""
        self.action_history.append(action)
        self.reasoning_history.append(action.get("reasoning", ""))
        self.step_count += 1
        if not valid:
            self.error_count += 1

    def get_trajectory(self) -> List[Tuple[Dict, Dict, str]]:
        """Return list of (observation, action, reasoning) tuples."""
        result = []
        n = min(len(self.observation_history), len(self.action_history))
        for i in range(n):
            result.append((
                self.observation_history[i],
                self.action_history[i],
                self.reasoning_history[i] if i < len(self.reasoning_history) else "",
            ))
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Return internal agent metrics."""
        return {
            "steps_taken": self.step_count,
            "error_count": self.error_count,
            "context_size": len(self.context_buffer),
            "error_rate": (
                self.error_count / self.step_count if self.step_count > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Rule-based helpers for subclasses
    # ------------------------------------------------------------------

    def _pick_action_type(self, valid_actions: List[str]) -> str:
        """Pick an action type from the valid set (with noise)."""
        if not valid_actions:
            return "search"
        # With probability proportional to position, pick a valid action
        return self._rng.choice(valid_actions)

    def _build_reasoning(self, goal: str, task_type: str, step: int) -> str:
        """Build a reasoning string referencing the goal."""
        goal_words = goal.split()
        # Pick some goal words to include in reasoning (simulate goal-awareness)
        n_words = max(2, len(goal_words) // 3)
        sampled_words = self._rng.sample(goal_words, min(n_words, len(goal_words)))
        return (
            f"Step {step}: Performing {task_type.lower()} to investigate "
            f"{' '.join(sampled_words)} as part of the research plan."
        )

    def _apply_degradation(
        self,
        base_value: float,
        step: int,
        slope: float,
        noise_std: float,
    ) -> float:
        """Apply linear degradation with Gaussian noise."""
        degraded = base_value + slope * step
        noise = self._rng.gauss(0, noise_std)
        return max(0.0, min(1.0, degraded + noise))
