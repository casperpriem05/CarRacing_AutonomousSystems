"""Generate ghost data one-at-a-time in separate processes to avoid Box2D segfaults."""

import json
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GHOST_DIR = os.path.join(_PROJECT_ROOT, "results", "ghosts")
BASE_SEED = 42
NUM_MAPS = 10

AGENTS = [
    ("random", "Random Baseline", "random_baseline"),
    ("cautious", "Cautious (My Sister)", "cautious_my_sister"),
    ("verstappen", "Max Verstappen", "max_verstappen"),
    ("traction", "Traction Focused", "traction_focused"),
    ("motogp", "MotoGP", "motogp"),
    ("rain", "Rain Rider", "rain_rider"),
    ("linehunter", "Line Hunter", "line_hunter"),
]

WORKER = '''
import json, os, sys, math
sys.path.insert(0, os.getcwd())
import config
from agents import ALL_AGENTS
from environment import ObservationProcessor
import gymnasium as gym
import numpy as np

agent_key = sys.argv[1]
seed = int(sys.argv[2])
out_path = sys.argv[3]

AGENT_MAP = {
    "random": 0, "cautious": 1, "verstappen": 2,
    "traction": 3, "motogp": 4, "rain": 5, "linehunter": 6,
}
agent = ALL_AGENTS[AGENT_MAP[agent_key]]()

env = gym.make(config.ENV_ID, continuous=False, render_mode=None, max_episode_steps=config.MAX_EPISODE_STEPS)
obs, info = env.reset(seed=seed)
processor = ObservationProcessor()
processor.reset()
agent.reset()

steps_data = []
total_reward = 0.0
done = False
while not done:
    features = processor.process(obs)
    action = agent.act(features)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    if hasattr(agent, "observe_reward"):
        agent.observe_reward(reward)
    car = env.unwrapped.car
    cx, cy, ca = float(car.hull.position[0]), float(car.hull.position[1]), float(car.hull.angle)
    steps_data.append({"total_reward": round(total_reward, 2), "on_track": features["on_track"], "speed": round(features["speed"], 3), "car_x": round(cx, 3), "car_y": round(cy, 3), "car_angle": round(ca, 4)})

env.close()
data = {"agent": agent.name, "seed": seed, "total_steps": len(steps_data), "final_reward": round(total_reward, 2), "steps": steps_data}
with open(out_path, "w") as f:
    json.dump(data, f)
print(f"reward={total_reward:.1f}")
'''


def main():
    os.makedirs(GHOST_DIR, exist_ok=True)
    worker_path = os.path.join(GHOST_DIR, "_worker.py")
    with open(worker_path, "w") as f:
        f.write(WORKER)

    total = len(AGENTS) * NUM_MAPS
    done_count = 0

    for agent_key, agent_name, file_prefix in AGENTS:
        for i in range(NUM_MAPS):
            seed = BASE_SEED + i
            out_path = os.path.join(GHOST_DIR, f"{file_prefix}_seed{seed}.json")
            done_count += 1

            if os.path.exists(out_path):
                print(f"  [{done_count}/{total}] {agent_name} seed {seed} — cached")
                continue

            print(f"  [{done_count}/{total}] {agent_name} seed {seed} — running...", end="", flush=True)
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, worker_path, agent_key, str(seed), out_path],
                capture_output=True, text=True, timeout=120,
                cwd=_PROJECT_ROOT,
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f" done ({elapsed:.1f}s, {result.stdout.strip()})")
            else:
                print(f" FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"    {result.stderr[:200]}")

    if os.path.exists(worker_path):
        os.remove(worker_path)
    print(f"\n  Done! {total} ghost files ready.")


if __name__ == "__main__":
    main()
