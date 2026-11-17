#!/usr/bin/env python3
"""
Generate publication-quality figures for the Planning Quality Degradation Study.

Reads from results/ CSV files and saves figures to results/figures/.

Author: Priyanshu Raj (MDE2026006)
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "ReAct": "#E63946",        # red
    "Reflexion": "#2A9D8F",    # teal
    "Hierarchical": "#264653", # dark blue-green
}
MARKERS = {"ReAct": "o", "Reflexion": "s", "Hierarchical": "^"}
ABLATION_COLORS = {
    "Reflexion_Full": "#2A9D8F",
    "Reflexion_NoMemory": "#E9C46A",
    "Reflexion_NoReflection": "#F4A261",
    "ReAct_Baseline": "#E63946",
}
ABLATION_LABELS = {
    "Reflexion_Full": "Reflexion (Full)",
    "Reflexion_NoMemory": "Reflexion (No Memory)",
    "Reflexion_NoReflection": "Reflexion (No Reflection)",
    "ReAct_Baseline": "ReAct (Baseline)",
}


def _ci(group: pd.Series):
    """Return (mean, sem) for a grouped Series."""
    return group.mean(), group.sem()


def load_data(results_dir: str):
    """Load all CSV results files."""
    main_df = pd.read_csv(os.path.join(results_dir, "experiment_results.csv"))
    abl_df = pd.read_csv(os.path.join(results_dir, "ablation_results.csv"))
    err_df = pd.read_csv(os.path.join(results_dir, "error_analysis.csv"))
    return main_df, abl_df, err_df


# ---------------------------------------------------------------------------
# Figure 1 — Degradation Curves (Primary)
# ---------------------------------------------------------------------------
def fig_degradation_curves(main_df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for agent in ["ReAct", "Reflexion", "Hierarchical"]:
        sub = main_df[main_df["agent"] == agent]
        grouped = sub.groupby("horizon_length")["goal_alignment_score"]
        horizons = sorted(sub["horizon_length"].unique())
        means = [grouped.get_group(h).mean() for h in horizons]
        sems  = [grouped.get_group(h).sem()  for h in horizons]

        ax.errorbar(
            horizons, means, yerr=[1.96 * s for s in sems],
            label=agent, color=COLORS[agent], marker=MARKERS[agent],
            linewidth=1.8, markersize=5, capsize=3,
        )

    ax.set_xlabel("Horizon Length (steps)")
    ax.set_ylabel("Goal Alignment Score")
    ax.set_title("Planning Quality Degradation by Architecture")
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="_threshold")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"degradation_curves.{ext}"))
    plt.close(fig)
    print("  ✓ degradation_curves")


# ---------------------------------------------------------------------------
# Figure 2 — All Metrics Comparison (2×3 grid)
# ---------------------------------------------------------------------------
def fig_all_metrics(main_df: pd.DataFrame, out_dir: str) -> None:
    metrics = [
        ("goal_alignment_score",        "Goal Alignment Score"),
        ("plan_validity_rate",          "Plan Validity Rate"),
        ("context_utilization_score",   "Context Utilization"),
        ("error_accumulation_rate",     "Error Accum. Rate"),
        ("task_completion",             "Task Completion Rate"),
        ("recovery_rate",               "Recovery Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes_flat = axes.flatten()

    for ax, (col, title) in zip(axes_flat, metrics):
        for agent in ["ReAct", "Reflexion", "Hierarchical"]:
            sub = main_df[main_df["agent"] == agent]
            grouped = sub.groupby("horizon_length")[col]
            horizons = sorted(sub["horizon_length"].unique())
            means = [grouped.get_group(h).mean() for h in horizons]
            sems  = [grouped.get_group(h).sem()  for h in horizons]
            ax.errorbar(
                horizons, means, yerr=[1.96 * s for s in sems],
                label=agent, color=COLORS[agent],
                marker=MARKERS[agent], linewidth=1.5, markersize=4, capsize=2,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Horizon", fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.grid(True, alpha=0.25, linestyle="--")

    # Shared legend
    handles = [
        mpatches.Patch(color=COLORS[a], label=a)
        for a in ["ReAct", "Reflexion", "Hierarchical"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("All Metrics: Planning Quality by Agent and Horizon", y=1.02)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"all_metrics_comparison.{ext}"))
    plt.close(fig)
    print("  ✓ all_metrics_comparison")


# ---------------------------------------------------------------------------
# Figure 3 — Error Analysis (Stacked Bar)
# ---------------------------------------------------------------------------
def fig_error_analysis(err_df: pd.DataFrame, out_dir: str) -> None:
    error_types = err_df["error_type"].unique()
    n_agents = 3
    agents = ["ReAct", "Reflexion", "Hierarchical"]

    # Use horizon 30 for the bar chart
    h30 = err_df[err_df["horizon_length"] == 30]
    agent_data = {ag: {} for ag in agents}
    for ag in agents:
        sub = h30[h30["agent"] == ag]
        for _, row in sub.iterrows():
            agent_data[ag][row["error_type"]] = row["count"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(agents))
    width = 0.6
    err_colors = ["#E63946", "#F4A261", "#E9C46A", "#2A9D8F", "#264653"]

    bottoms = np.zeros(len(agents))
    for et, color in zip(error_types, err_colors):
        vals = [agent_data[ag].get(et, 0) for ag in agents]
        ax.bar(x, vals, width, bottom=bottoms, label=et.replace("_", " ").title(),
               color=color, edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Error Count (Horizon = 30)")
    ax.set_title("Error Type Distribution at Horizon 30")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"error_analysis.{ext}"))
    plt.close(fig)
    print("  ✓ error_analysis")


# ---------------------------------------------------------------------------
# Figure 4 — Ablation Study
# ---------------------------------------------------------------------------
def fig_ablation(abl_df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    variants = ["Reflexion_Full", "Reflexion_NoMemory",
                "Reflexion_NoReflection", "ReAct_Baseline"]
    styles = ["-", "--", "-.", ":"]

    for variant, ls in zip(variants, styles):
        sub = abl_df[abl_df["variant"] == variant]
        grouped = sub.groupby("horizon_length")["goal_alignment_score"]
        horizons = sorted(sub["horizon_length"].unique())
        means = [grouped.get_group(h).mean() for h in horizons]
        sems  = [grouped.get_group(h).sem()  for h in horizons]
        ax.errorbar(
            horizons, means, yerr=[1.96 * s for s in sems],
            label=ABLATION_LABELS[variant],
            color=ABLATION_COLORS[variant],
            linestyle=ls, marker="o", linewidth=1.8, markersize=4, capsize=2,
        )

    ax.set_xlabel("Horizon Length (steps)")
    ax.set_ylabel("Goal Alignment Score")
    ax.set_title("Reflexion Ablation Study")
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"ablation_reflexion.{ext}"))
    plt.close(fig)
    print("  ✓ ablation_reflexion")


# ---------------------------------------------------------------------------
# Figure 5 — Degradation Rate Heatmap
# ---------------------------------------------------------------------------
def fig_degradation_heatmap(main_df: pd.DataFrame, out_dir: str) -> None:
    metrics = [
        "goal_alignment_score", "plan_validity_rate",
        "context_utilization_score", "error_accumulation_rate", "recovery_rate",
    ]
    metric_labels = [
        "Goal Align.", "Plan Valid.", "Context Util.",
        "Error Accum.", "Recovery Rate",
    ]
    agents = ["ReAct", "Reflexion", "Hierarchical"]

    slopes = np.zeros((len(agents), len(metrics)))
    for i, ag in enumerate(agents):
        sub = main_df[main_df["agent"] == ag]
        for j, m in enumerate(metrics):
            means_by_h = sub.groupby("horizon_length")[m].mean()
            x = np.array(means_by_h.index, dtype=float)
            y = np.array(means_by_h.values, dtype=float)
            if len(x) >= 2:
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes[i, j] = round(float(slope) * 10, 4)  # per 10 steps

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(slopes, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.05)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    ax.set_title("Degradation Rate Heatmap (change per 10 steps)")
    plt.colorbar(im, ax=ax, label="Δ per 10 steps")

    # Annotate cells
    for i in range(len(agents)):
        for j in range(len(metric_labels)):
            ax.text(j, i, f"{slopes[i,j]:+.3f}", ha="center", va="center",
                    fontsize=8, color="black")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"degradation_rate_heatmap.{ext}"))
    plt.close(fig)
    print("  ✓ degradation_rate_heatmap")


# ---------------------------------------------------------------------------
# Figure 6 — Task Completion Rate
# ---------------------------------------------------------------------------
def fig_task_completion(main_df: pd.DataFrame, out_dir: str) -> None:
    horizons = sorted(main_df["horizon_length"].unique())
    agents = ["ReAct", "Reflexion", "Hierarchical"]
    x = np.arange(len(horizons))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, agent in enumerate(agents):
        sub = main_df[main_df["agent"] == agent]
        tc_means = sub.groupby("horizon_length")["task_completion"].mean()
        vals = [tc_means.get(h, 0) for h in horizons]
        ax.bar(x + k * width, vals, width, label=agent,
               color=COLORS[agent], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Horizon Length (steps)")
    ax.set_ylabel("Task Completion Rate")
    ax.set_title("Task Completion Rate by Agent and Horizon")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"task_completion_rate.{ext}"))
    plt.close(fig)
    print("  ✓ task_completion_rate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading data...")
    main_df, abl_df, err_df = load_data(results_dir)

    print("Generating figures:")
    fig_degradation_curves(main_df, fig_dir)
    fig_all_metrics(main_df, fig_dir)
    fig_error_analysis(err_df, fig_dir)
    fig_ablation(abl_df, fig_dir)
    fig_degradation_heatmap(main_df, fig_dir)
    fig_task_completion(main_df, fig_dir)

    print(f"\n✓ All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
