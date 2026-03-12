#!/usr/bin/env python3
"""Entry point: run agents on CarRacing-v3, collect metrics, generate plots."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Type

import cv2
import numpy as np

import config
from agents import ALL_AGENTS
from agents.base_agent import BaseAgent
from agents.baseline_random import RandomAgent
from agents.agent_cautious import CautiousCruiser
from agents.agent_apex import ApexRacer
from agents.agent_traction import TractionManager
from agents.agent_superbike import MotoGP
from agents.agent_rain import RainRider
from agents.agent_line_hunter import LineHunter
from environment import ObservationProcessor, make_env
from evaluation.metrics import EpisodeMetrics, build_results_dataframe, summary_table
from evaluation.visualize import generate_all_plots

AGENT_MAP: Dict[str, Type[BaseAgent]] = {
    "random": RandomAgent,
    "cautious": CautiousCruiser,
    "verstappen": ApexRacer,
    "traction": TractionManager,
    "motogp": MotoGP,
    "rain": RainRider,
    "linehunter": LineHunter,
}

COLOURS = {
    "Random Baseline": "\033[90m",
    "Cautious (My Sister)": "\033[34m",
    "Max Verstappen": "\033[33m",
    "Traction Focused": "\033[36m",
    "MotoGP": "\033[31m",
    "Rain Rider": "\033[35m",
    "Line Hunter": "\033[32m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rule-based agents on CarRacing-v3."
    )
    parser.add_argument("--agent", type=str, default=None,
                        choices=list(AGENT_MAP.keys()),
                        help="Run a single agent by name (default: run all).")
    parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES,
                        help=f"Number of episodes per agent (default: {config.NUM_EPISODES}).")
    parser.add_argument("--render", action="store_true",
                        help="Enable human-visible rendering (slower).")
    parser.add_argument("--seed", type=int, default=config.BASE_SEED,
                        help=f"Base random seed (default: {config.BASE_SEED}).")
    parser.add_argument("--record", action="store_true",
                        help="Record video of best episode per agent.")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show live dashboard with progress bars and metrics.")
    return parser.parse_args()


def _bar(value: float, max_val: float, width: int = 30, fill: str = "#", empty: str = ".") -> str:
    ratio = max(0.0, min(1.0, value / max(max_val, 1e-9)))
    filled = int(ratio * width)
    return fill * filled + empty * (width - filled)


def _reward_colour(reward: float) -> str:
    if reward > 100:
        return GREEN
    elif reward > 0:
        return YELLOW
    return RED


def _print_episode_result(
    agent_name: str, ep: int, total_eps: int,
    reward: float, completion: float, off_track: float,
    best_reward: float, dashboard: bool
) -> None:
    colour = COLOURS.get(agent_name, "")
    rc = _reward_colour(reward)
    star = " * NEW BEST!" if reward >= best_reward else ""

    if dashboard:
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()
        reward_bar = _bar(max(0, reward + 100), 200, width=15)
        comp_bar = _bar(completion, 100, width=10)
        print(
            f"  {colour}{agent_name:20s}{RESET} "
            f"Ep {ep:3d}/{total_eps} "
            f"| Reward: {rc}{reward:7.1f}{RESET} {reward_bar} "
            f"| Lap: {comp_bar} {completion:5.1f}% "
            f"| Off: {off_track:4.1f}%"
            f"{GREEN}{star}{RESET}"
        )
    else:
        print(
            f"  {agent_name:20s} | Episode {ep:3d}/{total_eps} "
            f"| Reward: {reward:7.1f} "
            f"| Completion: {completion:5.1f}% "
            f"| Off-track: {off_track:5.1f}%"
            f"{star}"
        )


def _overlay_hud(frame: np.ndarray, agent_name: str, episode: int,
                 step: int, reward: float, speed: float,
                 on_track: bool, action: int) -> np.ndarray:
    h, w = frame.shape[:2]
    big = cv2.resize(frame, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)

    action_names = {0: "COAST", 1: "LEFT", 2: "RIGHT", 3: "GAS", 4: "BRAKE"}
    action_colours = {0: (180, 180, 180), 1: (255, 200, 50), 2: (255, 200, 50),
                      3: (50, 255, 50), 4: (50, 50, 255)}

    overlay = big.copy()
    cv2.rectangle(overlay, (0, 0), (big.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, big, 0.4, 0, big)

    white = (255, 255, 255)
    cv2.putText(big, f"{agent_name} | Episode {episode}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1, cv2.LINE_AA)

    rc = (50, 255, 50) if reward > 0 else (50, 50, 255)
    cv2.putText(big, f"Reward: {reward:.1f}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 1, cv2.LINE_AA)
    cv2.putText(big, f"Step: {step}", (200, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)

    bar_x, bar_w = 310, 60
    cv2.rectangle(big, (bar_x, 30), (bar_x + bar_w, 50), (60, 60, 60), -1)
    cv2.rectangle(big, (bar_x, 30), (bar_x + int(speed * bar_w), 50), (50, 255, 50), -1)
    cv2.putText(big, "SPD", (bar_x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, white, 1, cv2.LINE_AA)

    act_name = action_names.get(action, "?")
    act_col = action_colours.get(action, white)
    cv2.putText(big, act_name, (big.shape[1] - 80, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, act_col, 2, cv2.LINE_AA)

    if not on_track:
        cv2.putText(big, "OFF TRACK!", (big.shape[1] // 2 - 60, big.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return big


def record_best_episode(agent: BaseAgent, seed: int, episode_num: int) -> None:
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    processor = ObservationProcessor()
    env = make_env(render=False, seed=seed)
    obs, info = env.reset(seed=seed)
    processor.reset()
    agent.reset()

    frames = []
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        features = processor.process(obs)
        action = agent.act(features)
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if hasattr(agent, "observe_reward"):
            agent.observe_reward(reward)

        frame_bgr = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2BGR)
        annotated = _overlay_hud(
            frame_bgr, agent.name, episode_num, step,
            total_reward, features["speed"], features["on_track"], action
        )
        frames.append(annotated)
        obs = obs_next

    env.close()

    if not frames:
        return

    safe_name = agent.name.replace(" ", "_").lower()
    video_path = os.path.join(config.VIDEO_DIR, f"best_{safe_name}.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, config.VIDEO_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def run_agent(
    agent: BaseAgent,
    num_episodes: int,
    render: bool,
    base_seed: int,
    dashboard: bool = False,
) -> List[Dict[str, float]]:
    processor = ObservationProcessor()
    results: List[Dict[str, float]] = []
    best_reward = -float("inf")

    for ep in range(num_episodes):
        seed = base_seed + ep

        if dashboard:
            colour = COLOURS.get(agent.name, "")
            progress = _bar(ep + 1, num_episodes, width=20)
            print(f"\r  {colour}{BOLD}{agent.name:20s}{RESET} [{progress}] "
                  f"Episode {ep + 1:3d}/{num_episodes}", end="", flush=True)

        try:
            env = make_env(render=render, seed=seed)
            obs, info = env.reset(seed=seed)
            processor.reset()
            agent.reset()

            metrics = EpisodeMetrics()
            done = False

            while not done:
                features = processor.process(obs)
                action = agent.act(features)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                metrics.step(reward, features)

                if hasattr(agent, "observe_reward"):
                    agent.observe_reward(reward)

            env.close()
            summary = metrics.summarise()
            results.append(summary)

            ep_reward = summary["total_reward"]
            if ep_reward > best_reward:
                best_reward = ep_reward

            _print_episode_result(
                agent.name, ep + 1, num_episodes,
                ep_reward, summary["tiles_visited_pct"],
                summary["off_track_pct"], best_reward, dashboard
            )

        except Exception as exc:
            print(f"  {agent.name:20s} | Episode {ep + 1:3d}/{num_episodes} "
                  f"| SKIPPED ({type(exc).__name__})")
            try:
                env.close()
            except Exception:
                pass

    return results


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    if args.agent:
        agents_to_run = [AGENT_MAP[args.agent]()]
    else:
        agents_to_run = [cls() for cls in ALL_AGENTS]

    all_results: Dict[str, List[Dict[str, float]]] = {}

    print(f"\n  {BOLD}CarRacing-v3 Agent Evaluation{RESET}")
    print(f"  {DIM}Episodes: {args.episodes} | Seed: {args.seed} | Agents: {len(agents_to_run)}{RESET}\n")

    best_episodes: Dict[str, int] = {}

    for agent in agents_to_run:
        colour = COLOURS.get(agent.name, "")
        print(f"\n  {colour}{BOLD}Evaluating: {agent.name}{RESET}")
        t0 = time.time()
        ep_results = run_agent(
            agent, num_episodes=args.episodes, render=args.render,
            base_seed=args.seed, dashboard=args.dashboard,
        )
        elapsed = time.time() - t0
        all_results[agent.name] = ep_results

        if ep_results:
            rewards = [r["total_reward"] for r in ep_results]
            rc = _reward_colour(np.mean(rewards))
            print(f"    Avg: {rc}{np.mean(rewards):.1f}{RESET} | "
                  f"Best: {GREEN}{max(rewards):.1f}{RESET} | "
                  f"Time: {elapsed:.1f}s\n")
            best_ep_idx = int(np.argmax(rewards))
            best_episodes[agent.name] = args.seed + best_ep_idx

    if args.record:
        for agent_cls in (ALL_AGENTS if not args.agent else [AGENT_MAP[args.agent]]):
            agent = agent_cls()
            if agent.name in best_episodes:
                seed = best_episodes[agent.name]
                record_best_episode(agent, seed, seed - args.seed + 1)

    df = build_results_dataframe(all_results)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    df.to_csv(config.RESULTS_CSV, index=False)

    summary = summary_table(df)
    generate_all_plots(df, summary)

    print(f"\n  {BOLD}LEADERBOARD{RESET}")
    agent_scores = []
    for agent_name, results in all_results.items():
        if results:
            avg = np.mean([r["total_reward"] for r in results])
            best = max([r["total_reward"] for r in results])
            comp = np.mean([r["tiles_visited_pct"] for r in results])
            agent_scores.append((agent_name, avg, best, comp))

    agent_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, avg, best, comp) in enumerate(agent_scores, 1):
        colour = COLOURS.get(name, "")
        medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}.get(rank, "     ")
        rc = _reward_colour(avg)
        print(f"  {medal} {colour}{BOLD}{name:20s}{RESET} "
              f"Avg: {rc}{avg:7.1f}{RESET}  "
              f"Best: {GREEN}{best:7.1f}{RESET}  "
              f"Completion: {comp:5.1f}%")

    print()


if __name__ == "__main__":
    main()
