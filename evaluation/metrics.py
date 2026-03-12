"""Metrics collection for CarRacing-v3 agent evaluation.

Tracks per-episode:
    - total_reward
    - tiles_visited_pct  (lap completion %)
    - survival_steps
    - off_track_pct
    - avg_speed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class EpisodeMetrics:
    """Accumulates metrics for a single episode."""

    total_reward: float = 0.0
    steps: int = 0
    off_track_steps: int = 0
    speed_samples: List[float] = field(default_factory=list)
    tiles_visited: int = 0

    def step(self, reward: float, features: Dict[str, Any]) -> None:
        """Update metrics with one environment step."""
        self.total_reward += reward
        self.steps += 1
        if not features.get("on_track", True):
            self.off_track_steps += 1
        self.speed_samples.append(features.get("speed", 0.0))

    def record_tiles(self, info: dict) -> None:
        """Extract tiles-visited from the environment info dict (if available)."""
        pass

    def summarise(self) -> Dict[str, float]:
        """Return a flat dict of final metric values."""
        return {
            "total_reward": round(self.total_reward, 2),
            "survival_steps": self.steps,
            "off_track_pct": round(
                100.0 * self.off_track_steps / max(self.steps, 1), 2
            ),
            "avg_speed": round(
                float(np.mean(self.speed_samples)) if self.speed_samples else 0.0, 4
            ),
            "tiles_visited_pct": round(
                _reward_to_completion(self.total_reward), 2
            ),
        }


def _reward_to_completion(total_reward: float) -> float:
    """Approximate lap completion % from cumulative reward.

    In CarRacing-v3 the maximum achievable reward for a full lap is ~900.
    Each visited tile gives about +1000/N where N is the number of tiles.
    We cap at 100%.
    """
    return float(np.clip(total_reward / 900.0 * 100.0, 0.0, 100.0))


def build_results_dataframe(
    all_results: Dict[str, List[Dict[str, float]]],
) -> pd.DataFrame:
    """Convert per-agent episode results into a tidy DataFrame.

    Args:
        all_results: ``{agent_name: [episode_summary, ...]}``

    Returns:
        DataFrame with columns:
            agent, episode, total_reward, survival_steps,
            off_track_pct, avg_speed, tiles_visited_pct
    """
    rows = []
    for agent_name, episodes in all_results.items():
        for i, ep in enumerate(episodes):
            rows.append({"agent": agent_name, "episode": i + 1, **ep})
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a mean +/- std summary grouped by agent."""
    grouped = df.groupby("agent").agg(
        reward_mean=("total_reward", "mean"),
        reward_std=("total_reward", "std"),
        completion_mean=("tiles_visited_pct", "mean"),
        completion_std=("tiles_visited_pct", "std"),
        survival_mean=("survival_steps", "mean"),
        off_track_mean=("off_track_pct", "mean"),
        speed_mean=("avg_speed", "mean"),
    )
    return grouped.round(2)
