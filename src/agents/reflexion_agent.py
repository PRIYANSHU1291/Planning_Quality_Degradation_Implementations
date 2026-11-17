"""
Reflexion Agent — ReAct + verbal reflection + episodic memory.

Based on: Shinn et al. (2023) "Reflexion: Language Agents with Verbal
Reinforcement Learning." NeurIPS 2023.

Improvements over ReAct:
- After each step, generates a verbal reflection on the action outcome
- Maintains an episodic memory buffer of past reflections
- Reflection slows degradation but memory fills up at long horizons

Degradation profile (rule-based, slower than ReAct):
    goal_alignment:      start=0.92, slope=-0.010/step, noise=0.025
    plan_validity:       start=0.93, slope=-0.009/step, noise=0.022
    context_utilization: start=0.88, slope=-0.012/step, noise=0.028
    error_rate:          start=0.06, slope=+0.013/step
    recovery_rate:       start=0.45, slope=-0.008/step

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from .base_agent import BaseAgent


class ReflexionAgent(BaseAgent):
    """
    Reflexion agent: ReAct loop augmented with verbal reflection and
    episodic memory. Slower to degrade than ReAct; memory buffer gives
    a periodic boost but eventually causes context overload.

    Args:
        seed: Random seed.
        memory_size: Maximum number of reflections to retain.
        use_reflection: If False, disable verbal reflection (ablation mode).
        use_memory: If False, disable episodic memory (ablation mode).
        reflection_boost_interval: Steps between reflection boosts.
        reflection_boost_amount: Metric improvement per boost.
    """

    REFLECTION_TEMPLATES = [
        "The last action was {valid}. Key insight: {goal_fragment}.",
        "Reflecting on step {step}: {valid}. Adjusting focus toward {goal_fragment}.",
        "Memory note: action {action} {valid}. Continuing toward {goal_fragment}.",
    ]

    def __init__(
        self,
        seed: int = 42,
        memory_size: int = 10,
        use_reflection: bool = True,
        use_memory: bool = True,
        reflection_boost_interval: int = 5,
        reflection_boost_amount: float = 0.03,
        # Degradation profile
        goal_align_start: float = 0.92,
        goal_align_slope: float = -0.010,
        goal_align_noise: float = 0.025,
        plan_valid_start: float = 0.93,
        plan_valid_slope: float = -0.009,
        plan_valid_noise: float = 0.022,
        ctx_util_start: float = 0.88,
        ctx_util_slope: float = -0.012,
        ctx_util_noise: float = 0.028,
        error_rate_start: float = 0.06,
        error_rate_slope: float = 0.013,
        error_rate_noise: float = 0.022,
        recovery_start: float = 0.45,
        recovery_slope: float = -0.008,
        recovery_noise: float = 0.035,
    ) -> None:
        super().__init__(name="Reflexion", seed=seed)
        self.memory_size = memory_size
        self.use_reflection = use_reflection
        self.use_memory = use_memory
        self.reflection_boost_interval = reflection_boost_interval
        self.reflection_boost_amount = reflection_boost_amount

        # Degradation profile
        self.goal_align_start = goal_align_start
        self.goal_align_slope = goal_align_slope
        self.goal_align_noise = goal_align_noise
        self.plan_valid_start = plan_valid_start
        self.plan_valid_slope = plan_valid_slope
        self.plan_valid_noise = plan_valid_noise
        self.ctx_util_start = ctx_util_start
        self.ctx_util_slope = ctx_util_slope
        self.ctx_util_noise = ctx_util_noise
        self.error_rate_start = error_rate_start
        self.error_rate_slope = error_rate_slope
        self.error_rate_noise = error_rate_noise
        self.recovery_start = recovery_start
        self.recovery_slope = recovery_slope
        self.recovery_noise = recovery_noise

        # Episodic memory
        self.memory: Deque[str] = deque(maxlen=memory_size)
        self.last_action: Optional[str] = None

    def reset(self) -> None:
        super().reset()
        self.memory = deque(maxlen=self.memory_size)
        self.last_action = None

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a ReAct step augmented with reflection and memory lookup.
        """
        step = observation.get("step", self.step_count)
        goal = observation.get("goal", "")
        valid_actions = observation.get("valid_actions", [])
        task_type = observation.get("task_type", "")

        # Build reasoning incorporating memory (if enabled)
        reasoning = self._build_reasoning_with_memory(goal, task_type, step, valid_actions)

        action_type = self._pick_action_type(valid_actions)

        action = {
            "type": action_type,
            "reasoning": reasoning,
            "target": f"{task_type.lower()}_target_{step}",
            "agent": self.name,
            "step": step,
            "_metrics": self._compute_step_metrics(step),
        }

        # Generate reflection on this action (if enabled)
        if self.use_reflection:
            reflection = self._reflect(goal, action_type, step, valid_actions)
            action["_reflection"] = reflection
            if self.use_memory:
                self.memory.append(reflection)

        self.last_action = action_type
        self.record_action(
            action,
            valid=(action_type in valid_actions) if valid_actions else True,
        )
        return action

    def _build_reasoning_with_memory(
        self,
        goal: str,
        task_type: str,
        step: int,
        valid_actions: List[str],
    ) -> str:
        """Build reasoning trace that optionally incorporates memory."""
        base = self._build_reasoning(goal, task_type, step + 1)
        if self.use_memory and self.memory:
            mem_summary = " | ".join(list(self.memory)[-2:])
            return f"{base} [Memory: {mem_summary[:120]}]"
        return base

    def _reflect(
        self,
        goal: str,
        action_type: str,
        step: int,
        valid_actions: List[str],
    ) -> str:
        """Generate a verbal reflection after taking an action."""
        is_valid = action_type in valid_actions
        valid_str = "successful" if is_valid else "suboptimal"
        goal_words = goal.split()
        fragment = " ".join(self._rng.sample(goal_words, min(3, len(goal_words))))

        template = self._rng.choice(self.REFLECTION_TEMPLATES)
        return template.format(
            valid=valid_str,
            step=step + 1,
            action=action_type,
            goal_fragment=fragment,
        )

    def _compute_step_metrics(self, step: int) -> Dict[str, float]:
        """Compute step metrics with reflection boost at intervals."""
        # Base degradation
        gas = self._apply_degradation(
            self.goal_align_start, step, self.goal_align_slope, self.goal_align_noise
        )
        pv = self._apply_degradation(
            self.plan_valid_start, step, self.plan_valid_slope, self.plan_valid_noise
        )
        cu = self._apply_degradation(
            self.ctx_util_start, step, self.ctx_util_slope, self.ctx_util_noise
        )
        er = min(1.0, max(0.0,
            self.error_rate_start + self.error_rate_slope * step
            + self._rng.gauss(0, self.error_rate_noise)
        ))
        rr = max(0.0, self._apply_degradation(
            self.recovery_start, step, self.recovery_slope, self.recovery_noise
        ))

        # Periodic reflection boost
        if (self.use_reflection and step > 0
                and step % self.reflection_boost_interval == 0):
            boost = self.reflection_boost_amount
            gas = min(1.0, gas + boost)
            pv = min(1.0, pv + boost)
            rr = min(1.0, rr + boost * 0.5)

        return {
            "goal_alignment": gas,
            "plan_validity": pv,
            "context_utilization": cu,
            "error_rate": er,
            "recovery_rate": rr,
        }
