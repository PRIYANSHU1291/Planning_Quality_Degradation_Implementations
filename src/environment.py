"""
Multi-Step Research Planning Benchmark (MSRPB) Environment.

A synthetic benchmark environment for evaluating agent planning quality
across configurable horizon lengths. Designed for the study:
"Planning-Quality Degradation Across Agent Architectures in Long-Horizon Research Tasks"

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TaskType(Enum):
    LITERATURE_SEARCH = "LITERATURE_SEARCH"
    EVIDENCE_EVALUATION = "EVIDENCE_EVALUATION"
    HYPOTHESIS_FORMATION = "HYPOTHESIS_FORMATION"
    EXPERIMENT_DESIGN = "EXPERIMENT_DESIGN"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    SYNTHESIS = "SYNTHESIS"


# Allowed action types per task type
VALID_ACTION_MAP: Dict[str, List[str]] = {
    TaskType.LITERATURE_SEARCH.value:    ["search", "filter", "retrieve", "bookmark"],
    TaskType.EVIDENCE_EVALUATION.value:  ["evaluate", "compare", "verify", "reject"],
    TaskType.HYPOTHESIS_FORMATION.value: ["hypothesize", "refine", "validate", "reject"],
    TaskType.EXPERIMENT_DESIGN.value:    ["design", "plan", "configure", "document"],
    TaskType.DATA_ANALYSIS.value:        ["analyze", "compute", "visualize", "interpret"],
    TaskType.SYNTHESIS.value:            ["synthesize", "summarize", "conclude", "draft"],
}

TASK_GOALS: List[str] = [
    "investigate the effect of context length on agent planning quality",
    "compare planning strategies for long-horizon research tasks",
    "identify failure modes in autonomous research agents",
    "evaluate the role of memory in multi-step reasoning",
    "analyze goal drift patterns in extended agent trajectories",
    "study error accumulation in sequential decision making",
    "assess replanning effectiveness in dynamic research environments",
    "examine tool selection accuracy across increasing task complexity",
    "measure context utilization efficiency in long-horizon agents",
    "quantify the benefit of hierarchical task decomposition",
]


@dataclass
class TaskConfig:
    """Configuration for a single MSRPB task instance."""
    task_id: int
    horizon_length: int
    goal: str
    task_sequence: List[str]  # sequence of TaskType values
    seed: int
    checkpoints: List[int]  # steps at which checkpoints occur


@dataclass
class StepInfo:
    """Information returned after each environment step."""
    step: int
    task_type: str
    observation: str
    valid_actions: List[str]
    goal: str
    context_size: int
    horizon_remaining: int


class MSRPBEnvironment:
    """
    Multi-Step Research Planning Benchmark Environment.

    Simulates research-like planning tasks with configurable horizon lengths.
    Fully deterministic given a random seed. No LLM calls required.

    Args:
        horizon_length: Total number of steps in the task.
        seed: Random seed for reproducibility.
        task_id: Task instance identifier (selects goal and task sequence).
    """

    def __init__(self, horizon_length: int, seed: int = 42, task_id: int = 0) -> None:
        self.horizon_length = horizon_length
        self.seed = seed
        self.task_id = task_id

        self._rng = random.Random(seed + task_id * 997 + horizon_length * 31)
        self._config: Optional[TaskConfig] = None

        # Runtime state
        self.current_step: int = 0
        self.errors_made: int = 0
        self.goal_drift: float = 0.0
        self.context_buffer: List[str] = []
        self.action_history: List[Dict[str, Any]] = []
        self.observation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        self._rng = random.Random(self.seed + self.task_id * 997 + self.horizon_length * 31)
        self._config = self._generate_config()

        self.current_step = 0
        self.errors_made = 0
        self.goal_drift = 0.0
        self.context_buffer = [f"Research goal: {self._config.goal}"]
        self.action_history = []
        self.observation_history = []

        obs = self._make_observation()
        self.observation_history.append(obs)
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Dict with keys 'type', 'reasoning', 'target'.

        Returns:
            (observation, reward, done, info)
        """
        if self._config is None:
            raise RuntimeError("Call reset() before step().")

        evaluation = self.evaluate_action(action)
        reward = self._compute_reward(evaluation)

        # Update state
        self.current_step += 1
        if not evaluation["is_valid"]:
            self.errors_made += 1
        self.goal_drift += max(0.0, 1.0 - evaluation["goal_alignment"]) * 0.1

        # Update context
        context_entry = (
            f"[Step {self.current_step}] "
            f"Action: {action.get('type', 'unknown')} | "
            f"Reasoning: {str(action.get('reasoning', ''))[:80]}"
        )
        self.context_buffer.append(context_entry)
        self.action_history.append(action)

        done = self.current_step >= self.horizon_length
        obs = self._make_observation()
        self.observation_history.append(obs)

        info = {
            "evaluation": evaluation,
            "reward": reward,
            "errors_so_far": self.errors_made,
            "goal_drift": round(self.goal_drift, 4),
            "at_checkpoint": self.current_step in self._config.checkpoints,
        }
        return obs, reward, done, info

    def get_state(self) -> Dict[str, Any]:
        """Return the current full state dict."""
        return {
            "task_id": self.task_id,
            "horizon_length": self.horizon_length,
            "current_step": self.current_step,
            "goal": self._config.goal if self._config else "",
            "errors_made": self.errors_made,
            "goal_drift": round(self.goal_drift, 4),
            "context_size": len(self.context_buffer),
            "steps_remaining": max(0, self.horizon_length - self.current_step),
        }

    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an action against scoring rubrics.

        Args:
            action: Dict with 'type', 'reasoning', 'target'.

        Returns:
            Dict with goal_alignment, validity, context_relevance, is_valid.
        """
        if self._config is None or self.current_step >= self.horizon_length:
            return {
                "goal_alignment": 0.0,
                "validity": 0.0,
                "context_relevance": 0.0,
                "is_valid": False,
            }

        current_task = self._config.task_sequence[self.current_step]
        valid_actions = VALID_ACTION_MAP.get(current_task, [])

        action_type = str(action.get("type", "")).lower()
        reasoning = str(action.get("reasoning", "")).lower()
        goal = self._config.goal.lower()

        # Validity: is the action type appropriate for the current task?
        is_valid = action_type in valid_actions

        # Goal alignment: keyword overlap between reasoning and goal
        goal_words = set(goal.split())
        reasoning_words = set(reasoning.split()) if reasoning else set()
        overlap = len(goal_words & reasoning_words)
        goal_alignment = min(1.0, overlap / max(len(goal_words), 1) + (0.3 if is_valid else 0.0))

        # Context relevance: how much of the context is referenced
        context_text = " ".join(self.context_buffer[-5:]).lower()
        context_words = set(context_text.split())
        ctx_overlap = len(reasoning_words & context_words)
        context_relevance = min(1.0, ctx_overlap / max(len(context_words), 1) * 5)

        return {
            "goal_alignment": round(min(goal_alignment, 1.0), 4),
            "validity": 1.0 if is_valid else 0.0,
            "context_relevance": round(context_relevance, 4),
            "is_valid": is_valid,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_config(self) -> TaskConfig:
        """Generate a task configuration from the random seed."""
        goal_idx = self._rng.randint(0, len(TASK_GOALS) - 1)
        goal = TASK_GOALS[goal_idx]

        # Build task sequence: mix of task types proportional to horizon
        all_types = list(TaskType)
        task_seq = []
        for i in range(self.horizon_length):
            # Natural research flow: search -> evaluate -> hypothesize -> design -> analyze -> synthesize
            phase = int(i / max(self.horizon_length - 1, 1) * (len(all_types) - 1))
            jitter = self._rng.randint(-1, 1)
            idx = max(0, min(len(all_types) - 1, phase + jitter))
            task_seq.append(all_types[idx].value)

        checkpoints = list(range(4, self.horizon_length, 5))

        return TaskConfig(
            task_id=self.task_id,
            horizon_length=self.horizon_length,
            goal=goal,
            task_sequence=task_seq,
            seed=self.seed,
            checkpoints=checkpoints,
        )

    def _make_observation(self) -> Dict[str, Any]:
        """Build the current observation dict."""
        if self._config is None:
            return {}

        step = min(self.current_step, self.horizon_length - 1)
        current_task = self._config.task_sequence[step]
        valid_actions = VALID_ACTION_MAP.get(current_task, [])

        obs_text = (
            f"Step {self.current_step + 1}/{self.horizon_length}. "
            f"Task: {current_task}. "
            f"Goal: {self._config.goal}. "
            f"Valid actions: {valid_actions}. "
            f"Context size: {len(self.context_buffer)} entries."
        )

        return {
            "text": obs_text,
            "step": self.current_step,
            "task_type": current_task,
            "valid_actions": valid_actions,
            "goal": self._config.goal,
            "context_size": len(self.context_buffer),
            "horizon_remaining": self.horizon_length - self.current_step,
            "errors_so_far": self.errors_made,
        }

    def _compute_reward(self, evaluation: Dict[str, Any]) -> float:
        """Compute scalar reward from evaluation scores."""
        weights = {"goal_alignment": 0.5, "validity": 0.3, "context_relevance": 0.2}
        return round(
            weights["goal_alignment"] * evaluation["goal_alignment"]
            + weights["validity"] * evaluation["validity"]
            + weights["context_relevance"] * evaluation["context_relevance"],
            4,
        )


def generate_task_suite(
    horizon_lengths: List[int],
    n_tasks_per_horizon: int,
    seed: int = 42,
) -> List[TaskConfig]:
    """
    Generate a full task suite for experiments.

    Args:
        horizon_lengths: List of horizon lengths to include.
        n_tasks_per_horizon: Number of tasks per horizon length.
        seed: Master random seed.

    Returns:
        List of TaskConfig objects.
    """
    configs: List[TaskConfig] = []
    task_id = 0
    for horizon in horizon_lengths:
        for i in range(n_tasks_per_horizon):
            env = MSRPBEnvironment(horizon_length=horizon, seed=seed + i, task_id=task_id)
            env.reset()
            if env._config:
                configs.append(env._config)
            task_id += 1
    return configs
