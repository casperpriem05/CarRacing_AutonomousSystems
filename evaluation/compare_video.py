"""Generate a side-by-side comparison video of all agents' best episodes.

Records each agent one at a time, stores per-frame tiles,
then composites them into a single grid video.

Usage:
    python compare_video.py
    python compare_video.py --tile-size 320
    python compare_video.py --fps 30
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

import config
from agents import ALL_AGENTS
from agents.base_agent import BaseAgent
from environment import ObservationProcessor, make_env

AGENT_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Random Baseline":      (150, 150, 150),
    "Cautious (My Sister)": (255, 180, 50),
    "Max Verstappen":       (50, 220, 255),
    "Traction Focused":     (230, 200, 50),
    "MotoGP":               (60, 60, 255),
    "Rain Rider":           (200, 80, 220),
    "Line Hunter":          (80, 220, 80),
}

ACTION_NAMES = {0: "COAST", 1: "LEFT", 2: "RIGHT", 3: "GAS", 4: "BRAKE"}
ACTION_COLOURS = {
    0: (180, 180, 180), 1: (50, 200, 255), 2: (50, 200, 255),
    3: (50, 255, 50), 4: (50, 120, 255),
}


def _best_seeds_from_csv(csv_path: str, base_seed: int) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    best = {}
    for agent_name, group in df.groupby("agent"):
        best_row = group.loc[group["total_reward"].idxmax()]
        best[agent_name] = base_seed + int(best_row["episode"]) - 1
    return best


def _annotate_tile(
    frame_rgb: np.ndarray, agent_name: str, step: int, reward: float,
    speed: float, on_track: bool, action: int,
    colour: Tuple[int, int, int], tile_size: int,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    tile = cv2.resize(bgr, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

    border = 3
    cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), colour, border)

    overlay = tile.copy()
    cv2.rectangle(overlay, (border, border), (tile_size - border, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, tile, 0.35, 0, tile)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tile, agent_name, (8, 20), font, 0.48, colour, 2, cv2.LINE_AA)
    cv2.putText(tile, agent_name, (8, 20), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    rc = (80, 255, 80) if reward > 0 else (80, 80, 255)
    cv2.putText(tile, f"R:{reward:+.0f}", (8, 42), font, 0.40, rc, 1, cv2.LINE_AA)
    cv2.putText(tile, f"T:{step}", (tile_size // 2 - 10, 42), font, 0.35,
                (200, 200, 200), 1, cv2.LINE_AA)

    act_name = ACTION_NAMES.get(action, "?")
    act_col = ACTION_COLOURS.get(action, (200, 200, 200))
    text_size = cv2.getTextSize(act_name, font, 0.45, 1)[0]
    cv2.putText(tile, act_name, (tile_size - text_size[0] - 8, tile_size - 10),
                font, 0.45, act_col, 1, cv2.LINE_AA)

    bar_x, bar_y = 8, tile_size - 18
    bar_w, bar_h = 60, 8
    cv2.rectangle(tile, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(tile, (bar_x, bar_y), (bar_x + int(speed * bar_w), bar_y + bar_h), (50, 255, 50), -1)
    cv2.putText(tile, "SPD", (bar_x, bar_y - 2), font, 0.28, (180, 180, 180), 1, cv2.LINE_AA)

    if not on_track:
        cv2.putText(tile, "OFF TRACK", (tile_size // 2 - 45, tile_size // 2),
                    font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return tile


def _finished_tile(
    agent_name: str, reward: float, colour: Tuple[int, int, int], tile_size: int
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if reward > 100:
        bg = np.full((tile_size, tile_size, 3), 30, dtype=np.uint8)
        cv2.rectangle(bg, (0, 0), (tile_size - 1, tile_size - 1), colour, 3)
        cv2.putText(bg, agent_name, (8, 20), font, 0.48, colour, 1, cv2.LINE_AA)
        cv2.putText(bg, "LAP COMPLETE", (tile_size // 2 - 75, tile_size // 2 - 10),
                    font, 0.6, (80, 255, 80), 2, cv2.LINE_AA)
        rc = (80, 255, 80)
    else:
        bg = np.full((tile_size, tile_size, 3), 25, dtype=np.uint8)
        cv2.rectangle(bg, (0, 0), (tile_size - 1, tile_size - 1), colour, 3)
        cv2.putText(bg, agent_name, (8, 20), font, 0.48, colour, 1, cv2.LINE_AA)
        cv2.putText(bg, "CRASHED", (tile_size // 2 - 45, tile_size // 2 - 10),
                    font, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
        rc = (80, 80, 255)
    cv2.putText(bg, f"R:{reward:+.0f}", (tile_size // 2 - 30, tile_size // 2 + 20),
                font, 0.5, rc, 1, cv2.LINE_AA)
    return bg


def _build_grid(tiles: List[np.ndarray], tile_size: int, cols: int, padding: int = 4) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    grid_w = cols * tile_size + (cols + 1) * padding
    grid_h = rows * tile_size + (rows + 1) * padding
    grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = padding + r * (tile_size + padding)
        x = padding + c * (tile_size + padding)
        grid[y:y + tile_size, x:x + tile_size] = tile

    return grid


def _add_title_bar(grid: np.ndarray, text: str, height: int = 40) -> np.ndarray:
    w = grid.shape[1]
    bar = np.full((height, w, 3), 20, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(bar, text, (x, height - 12), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, grid])


def _record_single_agent(
    agent: BaseAgent, seed: int, tile_size: int, max_steps: int,
) -> Tuple[List[np.ndarray], float]:
    name = agent.name
    colour = AGENT_COLOURS.get(name, (200, 200, 200))
    processor = ObservationProcessor()
    env = make_env(render=False, seed=seed)
    obs, info = env.reset(seed=seed)
    processor.reset()
    agent.reset()

    tiles: List[np.ndarray] = []
    total_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        features = processor.process(obs)
        action = agent.act(features)
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if hasattr(agent, "observe_reward"):
            agent.observe_reward(reward)

        tile = _annotate_tile(
            obs, name, step, total_reward,
            features["speed"], features["on_track"], action,
            colour, tile_size,
        )
        tiles.append(tile)
        obs = obs_next

    env.close()
    return tiles, total_reward


def generate_comparison_video(
    best_seeds: Dict[str, int],
    output_path: str,
    tile_size: int = 320,
    fps: int = 15,
) -> None:
    agents_order: List[Tuple[type, int, str]] = []
    for cls in ALL_AGENTS:
        agent = cls()
        if agent.name in best_seeds:
            agents_order.append((cls, best_seeds[agent.name], agent.name))

    if not agents_order:
        return

    n = len(agents_order)
    cols = 4 if n > 6 else (3 if n > 4 else n)
    max_steps = config.MAX_EPISODE_STEPS

    all_agent_tiles: List[List[np.ndarray]] = []
    agent_names: List[str] = []
    agent_rewards: List[float] = []

    for cls, seed, name in agents_order:
        print(f"  Recording {name} (seed={seed})...", end=" ", flush=True)
        agent = cls()
        tiles, reward = _record_single_agent(agent, seed, tile_size, max_steps)
        all_agent_tiles.append(tiles)
        agent_names.append(name)
        agent_rewards.append(reward)
        print(f"{len(tiles)} frames, reward={reward:+.1f}")

    max_frames = max(len(t) for t in all_agent_tiles)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    sample_tiles = [all_agent_tiles[i][0] for i in range(n)]
    sample_grid = _add_title_bar(_build_grid(sample_tiles, tile_size, cols), "")
    h, w = sample_grid.shape[:2]

    codecs_to_try = [("avc1", "mp4"), ("mp4v", "mp4")]
    writer = None
    for codec, _ in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None:
        print("  No suitable video codec found")
        return

    for frame_idx in range(max_frames):
        tiles = []
        for i in range(n):
            if frame_idx < len(all_agent_tiles[i]):
                tiles.append(all_agent_tiles[i][frame_idx])
            else:
                colour = AGENT_COLOURS.get(agent_names[i], (200, 200, 200))
                tiles.append(_finished_tile(agent_names[i], agent_rewards[i], colour, tile_size))

        grid = _build_grid(tiles, tile_size, cols)
        grid = _add_title_bar(grid, f"CarRacing-v3 Agent Comparison  |  Step {frame_idx + 1}")
        writer.write(grid)

    writer.release()

    duration = max_frames / fps
    print(f"\n  Video saved: {output_path}")
    print(f"  {w}x{h}, {max_frames} frames @ {fps}fps ({duration:.0f}s)\n")

    ranked = sorted(zip(agent_names, agent_rewards), key=lambda x: x[1], reverse=True)
    for rank, (name, rew) in enumerate(ranked, 1):
        medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        print(f"  [{medal}] {name:<20s}  {rew:+.1f}")


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison video.")
    parser.add_argument("--seed", type=int, default=config.BASE_SEED)
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    csv_path = config.RESULTS_CSV
    output = args.output or os.path.join(config.VIDEO_DIR, "comparison.mp4")

    if not os.path.exists(csv_path):
        print(f"  No results.csv found. Run main.py first.")
        sys.exit(1)

    best_seeds = _best_seeds_from_csv(csv_path, args.seed)
    generate_comparison_video(best_seeds, output, tile_size=args.tile_size, fps=args.fps)


if __name__ == "__main__":
    main()
