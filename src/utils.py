"""
Utility functions for the Planning Quality Degradation Study.

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def set_all_seeds(seed: int) -> None:
    """Set numpy and random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Return current timestamp in ISO 8601 format."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def create_experiment_id(agent_name: str, horizon: int, seed: int) -> str:
    """Create a unique experiment identifier string."""
    return f"{agent_name}__h{horizon:02d}__s{seed}"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file. Falls back to empty dict if yaml unavailable."""
    if not HAS_YAML:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save a config dict to a YAML file."""
    ensure_dir(os.path.dirname(path))
    if not HAS_YAML:
        # Fallback: write as plain text
        with open(path, "w") as f:
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
        return
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def compute_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.

    Args:
        values: List of numeric values.
        confidence: Confidence level (default 0.95).

    Returns:
        (mean, lower_bound, upper_bound)
    """
    from scipy import stats

    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    se = float(stats.sem(arr))
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return round(mean, 4), round(mean - h, 4), round(mean + h, 4)


def format_metric(value: float, std: Optional[float] = None) -> str:
    """Format a metric value as '0.XXX' or '0.XXX ± 0.XXX'."""
    if std is not None:
        return f"{value:.3f} ± {std:.3f}"
    return f"{value:.3f}"


def format_results_table(
    results_df: pd.DataFrame,
    metrics: List[str],
    index_col: str = "horizon_length",
    group_col: str = "agent",
) -> str:
    """
    Format results as a publication-quality markdown table.

    Args:
        results_df: DataFrame with experimental results.
        metrics: Metric column names to include.
        index_col: Row index column.
        group_col: Column used for grouping.

    Returns:
        Markdown table string.
    """
    summary = results_df.groupby([index_col, group_col])[metrics].agg(
        ["mean", "std"]
    ).round(3)

    lines = []
    agents = sorted(results_df[group_col].unique())
    metric = metrics[0]  # Use primary metric for table

    # Header
    header = f"| {index_col.replace('_', ' ').title()} |"
    for ag in agents:
        header += f" {ag} |"
    lines.append(header)
    lines.append("|" + "---|" * (len(agents) + 1))

    # Rows
    for idx_val in sorted(results_df[index_col].unique()):
        row = f"| {idx_val} |"
        for ag in agents:
            try:
                mean = summary.loc[(idx_val, ag), (metric, "mean")]
                std = summary.loc[(idx_val, ag), (metric, "std")]
                row += f" {mean:.3f}±{std:.3f} |"
            except KeyError:
                row += " — |"
        lines.append(row)

    return "\n".join(lines)
