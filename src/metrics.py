"""
Evaluation metrics for the Planning Quality Degradation Study.

Provides composite planning quality metrics, trajectory evaluation,
and statistical comparison utilities.

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class PlanningQualityMetrics:
    """
    Collection of planning quality metrics for long-horizon agent evaluation.

    All metrics return values in [0, 1] where higher is better,
    except error_accumulation_rate where lower is better (we invert it
    to 1 - error_rate so higher is still better).
    """

    @staticmethod
    def compute_goal_alignment(
        action_reasoning: str, original_goal: str
    ) -> float:
        """
        Compute goal alignment as word-overlap between reasoning and goal.

        Args:
            action_reasoning: The reasoning string from the agent.
            original_goal: The original research goal string.

        Returns:
            Float in [0, 1].
        """
        if not original_goal:
            return 0.0
        goal_words = set(original_goal.lower().split())
        reason_words = set(action_reasoning.lower().split())
        if not goal_words:
            return 0.0
        return min(1.0, len(goal_words & reason_words) / len(goal_words))

    @staticmethod
    def compute_plan_validity(
        action: Dict[str, Any],
        valid_actions: List[str],
        state: Dict[str, Any],
    ) -> float:
        """
        Compute plan validity: 1 if action type is in valid set, else 0.

        Args:
            action: Action dict with 'type' key.
            valid_actions: List of valid action type strings.
            state: Current environment state dict.

        Returns:
            Float in {0.0, 1.0}.
        """
        if not valid_actions:
            return 1.0
        action_type = str(action.get("type", "")).lower()
        return 1.0 if action_type in valid_actions else 0.0

    @staticmethod
    def compute_context_utilization(
        used_context: List[str], available_context: List[str]
    ) -> float:
        """
        Compute context utilization as coverage of available context in use.

        Args:
            used_context: List of context items referenced in reasoning.
            available_context: Full context buffer.

        Returns:
            Float in [0, 1].
        """
        if not available_context:
            return 0.0
        used_set = set(" ".join(used_context).lower().split())
        available_set = set(" ".join(available_context).lower().split())
        if not available_set:
            return 0.0
        return min(1.0, len(used_set & available_set) / len(available_set))

    @staticmethod
    def compute_error_accumulation(
        errors: int, total_steps: int
    ) -> float:
        """
        Compute inverted error accumulation rate (1 - error_rate).

        Args:
            errors: Number of invalid/erroneous actions.
            total_steps: Total number of steps taken.

        Returns:
            Float in [0, 1] — higher means fewer errors.
        """
        if total_steps == 0:
            return 1.0
        error_rate = errors / total_steps
        return max(0.0, 1.0 - error_rate)

    @staticmethod
    def compute_task_completion(
        trajectory: List[Tuple], task_goal: str
    ) -> bool:
        """
        Estimate task completion from trajectory quality.

        Args:
            trajectory: List of (observation, action, reasoning) tuples.
            task_goal: The goal string.

        Returns:
            True if average goal alignment exceeds 0.5.
        """
        if not trajectory:
            return False
        alignments = []
        for obs, action, reasoning in trajectory:
            goal_words = set(task_goal.lower().split())
            reason_words = set(reasoning.lower().split())
            if goal_words:
                score = len(goal_words & reason_words) / len(goal_words)
                alignments.append(min(1.0, score))
        return float(np.mean(alignments)) > 0.5 if alignments else False

    @staticmethod
    def compute_recovery_rate(
        errors_detected: int, errors_recovered: int
    ) -> float:
        """
        Compute fraction of detected errors that were successfully recovered.

        Args:
            errors_detected: Number of errors the agent encountered.
            errors_recovered: Number of errors from which the agent recovered.

        Returns:
            Float in [0, 1].
        """
        if errors_detected == 0:
            return 1.0
        return min(1.0, errors_recovered / errors_detected)


def compute_degradation_slope(metric_values: List[float]) -> float:
    """
    Estimate the linear degradation rate from a series of metric values.

    Args:
        metric_values: Time-ordered list of metric measurements.

    Returns:
        Slope (change per step) from linear regression.
    """
    if len(metric_values) < 2:
        return 0.0
    x = np.arange(len(metric_values), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, metric_values)
    return float(slope)


def compute_planning_quality_index(metrics_dict: Dict[str, float]) -> float:
    """
    Compute a weighted composite Planning Quality Index (PQI).

    Weights reflect relative importance to research task success:
        goal_alignment: 40%
        plan_validity: 30%
        context_utilization: 15%
        error_accumulation (inverted): 10%
        recovery_rate: 5%

    Args:
        metrics_dict: Dict with metric names and [0,1] values.

    Returns:
        Weighted composite float in [0, 1].
    """
    weights = {
        "goal_alignment": 0.40,
        "plan_validity": 0.30,
        "context_utilization": 0.15,
        "error_accumulation": 0.10,
        "recovery_rate": 0.05,
    }
    total = 0.0
    for key, w in weights.items():
        total += w * float(metrics_dict.get(key, 0.0))
    return round(total, 4)


def statistical_comparison(
    results_a: List[float],
    results_b: List[float],
    metric: str = "",
) -> Dict[str, float]:
    """
    Run an independent-samples t-test and compute Cohen's d effect size.

    Args:
        results_a: Metric values for group A.
        results_b: Metric values for group B.
        metric: Metric name (for labelling only).

    Returns:
        Dict with t_stat, p_value, effect_size (Cohen's d),
        mean_a, mean_b, std_a, std_b.
    """
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d (pooled std)
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    cohen_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
    return {
        "metric": metric,
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "effect_size": round(float(cohen_d), 4),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "std_a": round(float(a.std(ddof=1)), 4),
        "std_b": round(float(b.std(ddof=1)), 4),
    }


class TrajectoryEvaluator:
    """Evaluates agent trajectories on the MSRPB environment."""

    def __init__(self) -> None:
        self.metrics = PlanningQualityMetrics()

    def evaluate_trajectory(
        self,
        agent: Any,
        environment: Any,
        task_config: Any,
    ) -> Dict[str, Any]:
        """
        Run one full trajectory and collect per-step metrics.

        Args:
            agent: An agent instance with act() method.
            environment: An MSRPBEnvironment instance.
            task_config: TaskConfig for the episode.

        Returns:
            Dict with 'steps' (list of per-step dicts) and 'summary'.
        """
        obs = environment.reset()
        agent.reset()

        step_records: List[Dict[str, Any]] = []
        done = False

        while not done:
            action = agent.act(obs)
            obs_next, reward, done, info = environment.step(action)

            # Collect metrics from agent's internal simulation
            agent_metrics = action.get("_metrics", {})
            step_records.append({
                "step": obs.get("step", 0),
                "task_type": obs.get("task_type", ""),
                "action_type": action.get("type", ""),
                "reward": reward,
                "valid": info["evaluation"]["is_valid"],
                **{k: round(v, 4) for k, v in agent_metrics.items()},
            })
            obs = obs_next

        # Summary stats
        df = pd.DataFrame(step_records)
        summary: Dict[str, Any] = {
            "horizon_length": task_config.horizon_length,
            "task_id": task_config.task_id,
            "total_errors": environment.errors_made,
            "total_steps": environment.current_step,
        }
        for col in ["goal_alignment", "plan_validity", "context_utilization",
                    "error_rate", "recovery_rate"]:
            if col in df.columns:
                summary[f"mean_{col}"] = round(float(df[col].mean()), 4)

        # Task completion heuristic
        mean_ga = summary.get("mean_goal_alignment", 0.0)
        mean_pv = summary.get("mean_plan_validity", 0.0)
        summary["task_completion"] = int((mean_ga + mean_pv) / 2 > 0.5)

        return {"steps": step_records, "summary": summary}

    def evaluate_suite(
        self,
        agent_class: type,
        environment_class: type,
        task_configs: List[Any],
        agent_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Evaluate an agent over a full task suite.

        Returns a DataFrame with one row per task.
        """
        agent_kwargs = agent_kwargs or {}
        rows = []
        for config in task_configs:
            env = environment_class(
                horizon_length=config.horizon_length,
                seed=config.seed,
                task_id=config.task_id,
            )
            agent = agent_class(**agent_kwargs)
            result = self.evaluate_trajectory(agent, env, config)
            rows.append(result["summary"])
        return pd.DataFrame(rows)

    def compute_degradation_curve(
        self, results_df: pd.DataFrame, metric_name: str
    ) -> Dict[str, float]:
        """
        Fit a linear degradation curve over horizon lengths.

        Args:
            results_df: DataFrame with 'horizon_length' and metric_name columns.
            metric_name: Column to analyse.

        Returns:
            Dict with slope, intercept, r_squared.
        """
        means = results_df.groupby("horizon_length")[metric_name].mean()
        x = np.array(means.index, dtype=float)
        y = np.array(means.values, dtype=float)
        slope, intercept, r, _, _ = stats.linregress(x, y)
        return {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r ** 2), 4),
        }
