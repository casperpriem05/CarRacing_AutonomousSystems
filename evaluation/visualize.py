"""Visualisation helpers — generate comparison plots and tables.

All figures are saved to the ``results/`` directory as PNG files.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid", palette="muted")

AGENT_ORDER: Optional[list] = None


def _ensure_dir() -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return config.RESULTS_DIR


def _agent_order(df: pd.DataFrame) -> list:
    global AGENT_ORDER
    if AGENT_ORDER is None:
        AGENT_ORDER = sorted(df["agent"].unique())
    return AGENT_ORDER


def plot_mean_reward(df: pd.DataFrame) -> None:
    """Bar chart of mean total reward per agent with std-dev error bars."""
    out = _ensure_dir()
    order = _agent_order(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="agent", y="total_reward", order=order,
                errorbar="sd", capsize=0.1, ax=ax)
    ax.set_title("Mean Total Reward per Agent", fontsize=14)
    ax.set_ylabel("Total Reward")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out, "mean_reward.png"), dpi=150)
    plt.close(fig)


def plot_reward_boxplot(df: pd.DataFrame) -> None:
    """Box plot showing the full reward distribution for each agent."""
    out = _ensure_dir()
    order = _agent_order(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="agent", y="total_reward", order=order, ax=ax)
    ax.set_title("Reward Distribution per Agent", fontsize=14)
    ax.set_ylabel("Total Reward")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out, "reward_boxplot.png"), dpi=150)
    plt.close(fig)


def plot_completion(df: pd.DataFrame) -> None:
    """Bar chart of mean tiles-visited / lap-completion percentage."""
    out = _ensure_dir()
    order = _agent_order(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="agent", y="tiles_visited_pct", order=order,
                errorbar="sd", capsize=0.1, ax=ax)
    ax.set_title("Mean Lap Completion (%) per Agent", fontsize=14)
    ax.set_ylabel("Completion %")
    ax.set_xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out, "completion.png"), dpi=150)
    plt.close(fig)


def plot_episode_rewards(df: pd.DataFrame) -> None:
    """Line plot of reward over episodes for each agent."""
    out = _ensure_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    for agent_name, group in df.groupby("agent"):
        group = group.sort_values("episode")
        ax.plot(group["episode"], group["total_reward"], label=agent_name, alpha=0.8)
    ax.set_title("Episode Rewards Over Time", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "episode_rewards.png"), dpi=150)
    plt.close(fig)


def print_and_save_table(summary: pd.DataFrame) -> None:
    """Print summary table to stdout and save as a PNG image."""
    out = _ensure_dir()
    print("\nAGENT COMPARISON TABLE")
    print(summary.to_string())
    print()

    fig, ax = plt.subplots(figsize=(12, 2 + 0.4 * len(summary)))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        rowLabels=summary.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title("Agent Comparison — Summary Statistics", fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "comparison_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar(summary: pd.DataFrame) -> None:
    """Radar chart comparing agents across multiple normalised metrics."""
    out = _ensure_dir()
    categories = ["reward_mean", "completion_mean", "survival_mean", "speed_mean"]
    labels = ["Reward", "Completion", "Survival", "Avg Speed"]
    agents = summary.index.tolist()

    values_raw = summary[categories].values.astype(float)
    mins = values_raw.min(axis=0)
    maxs = values_raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    values_norm = (values_raw - mins) / ranges

    if "off_track_mean" in summary.columns:
        ot = summary["off_track_mean"].values.astype(float)
        ot_norm = 1.0 - (ot - ot.min()) / max(ot.max() - ot.min(), 1e-9)
        values_norm = np.column_stack([values_norm, ot_norm])
        labels.append("On-Track %")

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for i, agent in enumerate(agents):
        vals = values_norm[i].tolist() + [values_norm[i][0]]
        ax.plot(angles, vals, linewidth=1.8, label=agent)
        ax.fill(angles, vals, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "radar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Run every visualisation and save results to disk."""
    plot_mean_reward(df)
    plot_reward_boxplot(df)
    plot_completion(df)
    plot_episode_rewards(df)
    print_and_save_table(summary)
    plot_radar(summary)
    print(f"All plots saved to '{config.RESULTS_DIR}/'")
