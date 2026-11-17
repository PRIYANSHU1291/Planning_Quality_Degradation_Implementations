"""
Hierarchical Planning Agent — Coarse-to-fine planning with subgoal decomposition.

Based on:
- Aghzal, Stein & Yao (2026) "Why Do LLM-based Web Agents Fail?
  A Hierarchical Planning Perspective." ACL 2026.
- Tan et al. (2026) "From Coarse to Fine: Self-Adaptive Hierarchical
  Planning for LLM Agents." ACL Findings 2026.

Key mechanisms:
- Two-level planning: coarse subgoals (every ~5 steps) + fine execution
- Selective context management: only retains relevant context per subgoal
- Replanning trigger: when any metric drops below threshold
- Subgoal boundaries act as "reset points" preventing error cascading

Degradation profile (best long-horizon performance):
    goal_alignment:      start=0.88, slope=-0.006/step, noise=0.022
    plan_validity:       start=0.90, slope=-0.005/step, noise=0.020
    context_utilization: start=0.90, slope=-0.007/step, noise=0.025
    error_rate:          start=0.07, slope=+0.008/step
    recovery_rate:       start=0.55, slope=-0.005/step

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent


class HierarchicalAgent(BaseAgent):
    """
    Hierarchical Planning agent with coarse-to-fine planning and
    selective context management. Best performance at long horizons.

    Args:
        seed: Random seed.
        subgoal_size: Number of steps per subgoal chunk.
        replan_threshold: Metric value below which replanning is triggered.
        subgoal_boost: Metric improvement at each subgoal boundary.
    """

    def __init__(
        self,
        seed: int = 42,
        subgoal_size: int = 5,
        replan_threshold: float = 0.50,
        subgoal_boost: float = 0.04,
        # Degradation profile
        goal_align_start: float = 0.88,
        goal_align_slope: float = -0.006,
        goal_align_noise: float = 0.022,
        plan_valid_start: float = 0.90,
        plan_valid_slope: float = -0.005,
        plan_valid_noise: float = 0.020,
        ctx_util_start: float = 0.90,
        ctx_util_slope: float = -0.007,
        ctx_util_noise: float = 0.025,
        error_rate_start: float = 0.07,
        error_rate_slope: float = 0.008,
        error_rate_noise: float = 0.020,
        recovery_start: float = 0.55,
        recovery_slope: float = -0.005,
        recovery_noise: float = 0.030,
    ) -> None:
        super().__init__(name="Hierarchical", seed=seed)
        self.subgoal_size = subgoal_size
        self.replan_threshold = replan_threshold
        self.subgoal_boost = subgoal_boost

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

        # Hierarchical state
        self.current_subgoal_idx: int = 0
        self.subgoal_context: List[str] = []
        self.replan_count: int = 0

    def reset(self) -> None:
        super().reset()
        self.current_subgoal_idx = 0
        self.subgoal_context = []
        self.replan_count = 0

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce an action based on hierarchical planning.
        Steps are grouped into subgoals; at each boundary, selective
        context compression and optional replanning occur.
        """
        step = observation.get("step", self.step_count)
        goal = observation.get("goal", "")
        valid_actions = observation.get("valid_actions", [])
        task_type = observation.get("task_type", "")

        # Check subgoal boundary
        at_boundary = (step > 0) and (step % self.subgoal_size == 0)
        if at_boundary:
            self.current_subgoal_idx += 1
            self._compress_context()

        reasoning = self._build_hierarchical_reasoning(goal, task_type, step, at_boundary)
        action_type = self._pick_action_type(valid_actions)

        metrics = self._compute_step_metrics(step, at_boundary)

        # Trigger replanning if any metric below threshold
        replanned = False
        if any(v < self.replan_threshold for k, v in metrics.items()
               if k not in ("error_rate",)):
            replanned = True
            self.replan_count += 1
            # Replanning recovers some quality
            metrics["goal_alignment"] = min(
                1.0, metrics["goal_alignment"] + self.subgoal_boost * 0.5
            )
            metrics["plan_validity"] = min(
                1.0, metrics["plan_validity"] + self.subgoal_boost * 0.5
            )

        action = {
            "type": action_type,
            "reasoning": reasoning,
            "target": f"{task_type.lower()}_subgoal{self.current_subgoal_idx}_step{step}",
            "agent": self.name,
            "step": step,
            "subgoal_idx": self.current_subgoal_idx,
            "at_boundary": at_boundary,
            "replanned": replanned,
            "_metrics": metrics,
        }

        self.record_action(
            action,
            valid=(action_type in valid_actions) if valid_actions else True,
        )
        return action

    def _compress_context(self) -> None:
        """Selective context compression at subgoal boundaries."""
        # Keep only the most recent subgoal-relevant context (simulate selective retention)
        if len(self.context_buffer) > self.subgoal_size * 2:
            # Retain goal + last subgoal context only
            goal_entries = [e for e in self.context_buffer if "goal:" in e.lower()]
            recent = self.context_buffer[-(self.subgoal_size):]
            self.context_buffer = goal_entries + recent

    def _build_hierarchical_reasoning(
        self,
        goal: str,
        task_type: str,
        step: int,
        at_boundary: bool,
    ) -> str:
        """Build reasoning trace that includes subgoal structure."""
        base = self._build_reasoning(goal, task_type, step + 1)
        subgoal_note = (
            f" [Subgoal {self.current_subgoal_idx + 1} started — context refreshed.]"
            if at_boundary
            else f" [Subgoal {self.current_subgoal_idx + 1}, step {step % self.subgoal_size + 1}]"
        )
        return base + subgoal_note

    def _compute_step_metrics(
        self, step: int, at_boundary: bool
    ) -> Dict[str, float]:
        """Compute step metrics with subgoal boundary boosts."""
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

        # Subgoal boundary: context compression gives metric boost
        if at_boundary:
            gas = min(1.0, gas + self.subgoal_boost)
            pv = min(1.0, pv + self.subgoal_boost)
            cu = min(1.0, cu + self.subgoal_boost * 1.5)  # context most improved
            rr = min(1.0, rr + self.subgoal_boost * 0.5)

        return {
            "goal_alignment": gas,
            "plan_validity": pv,
            "context_utilization": cu,
            "error_rate": er,
            "recovery_rate": rr,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics including hierarchical-specific stats."""
        base = super().get_metrics()
        base.update({
            "subgoal_count": self.current_subgoal_idx + 1,
            "replan_count": self.replan_count,
        })
        return base
