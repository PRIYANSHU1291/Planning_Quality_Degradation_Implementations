"""
ReAct Agent — Standard reason-then-act loop.

Based on: Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in
Language Models." ICLR 2023.

Degradation profile (rule-based simulation):
- No persistent memory across steps
- No self-reflection mechanism
- Quality degrades linearly with step count due to error accumulation

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base_agent import BaseAgent


class ReActAgent(BaseAgent):
    """
    ReAct agent: interleaves reasoning and acting without memory or reflection.

    Rule-based degradation parameters (inferred from literature):
        goal_alignment:      start=0.90, slope=-0.015/step, noise=0.030
        plan_validity:       start=0.92, slope=-0.012/step, noise=0.025
        context_utilization: start=0.85, slope=-0.018/step, noise=0.035
        error_rate:          start=0.05, slope=+0.020/step
        recovery_rate:       fixed ~0.15 (no recovery mechanism)
    """

    def __init__(
        self,
        seed: int = 42,
        goal_align_start: float = 0.90,
        goal_align_slope: float = -0.015,
        goal_align_noise: float = 0.030,
        plan_valid_start: float = 0.92,
        plan_valid_slope: float = -0.012,
        plan_valid_noise: float = 0.025,
        ctx_util_start: float = 0.85,
        ctx_util_slope: float = -0.018,
        ctx_util_noise: float = 0.035,
        error_rate_start: float = 0.05,
        error_rate_slope: float = 0.020,
        error_rate_noise: float = 0.025,
        recovery_rate: float = 0.15,
    ) -> None:
        super().__init__(name="ReAct", seed=seed)
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
        self.recovery_rate = recovery_rate

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a reason-then-act step (no memory, no reflection)."""
        step = observation.get("step", self.step_count)
        goal = observation.get("goal", "")
        valid_actions = observation.get("valid_actions", [])
        task_type = observation.get("task_type", "")

        reasoning = self._build_reasoning(goal, task_type, step + 1)
        action_type = self._pick_action_type(valid_actions)

        action = {
            "type": action_type,
            "reasoning": reasoning,
            "target": f"{task_type.lower()}_target_{step}",
            "agent": self.name,
            "step": step,
            "_metrics": self._compute_step_metrics(step),
        }

        self.record_action(
            action,
            valid=(action_type in valid_actions) if valid_actions else True,
        )
        return action

    def _compute_step_metrics(self, step: int) -> Dict[str, float]:
        """Compute simulated planning quality metrics for this step."""
        return {
            "goal_alignment": self._apply_degradation(
                self.goal_align_start, step, self.goal_align_slope, self.goal_align_noise
            ),
            "plan_validity": self._apply_degradation(
                self.plan_valid_start, step, self.plan_valid_slope, self.plan_valid_noise
            ),
            "context_utilization": self._apply_degradation(
                self.ctx_util_start, step, self.ctx_util_slope, self.ctx_util_noise
            ),
            "error_rate": min(1.0, max(0.0,
                self.error_rate_start + self.error_rate_slope * step
                + self._rng.gauss(0, self.error_rate_noise)
            )),
            "recovery_rate": self.recovery_rate,
        }
