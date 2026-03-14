
# CarRacing: Motorcycle-Inspired Rule-Based Agents

A university portfolio project for a **Rule-Based Systems** course. Seven agents (1 baseline + 6 expert) compete in the Gymnasium `CarRacing-v3` environment using only hand-crafted rules, no machine learning -> autonomous systems.

Every agent's decision logic is motivated by **real-world motorcycle riding dynamics**: braking before curves, managing traction, following the racing line, and adapting aggression on the fly.

<video src="results/videos/comparison.mp4" controls width="100%"></video>

## Agents

| Agent | Metaphor | Strategy |
|---|---|---|
| **Random Baseline** | None | Uniform random actions (performance floor) |
| **Cautious (My Sister)** | New rider | Safety-first: brakes early, gentle corrections, anticipatory mid-offset steering |
| **Max Verstappen** | Track-day rider | "Slow in, fast out" cornering via a 4-phase state machine |
| **Traction Focused** | Grip-conscious rider | Friction-circle traction budget limits simultaneous steering + gas |
| **MotoGP** | MotoGP rider | Three-tier steering with adaptive confidence and reward-based aggression scaling |
| **Rain Rider** | Wet-weather rider | Adaptive aggression that backs off after mistakes and pushes when confident |
| **Line Hunter** | Precision rider | Weighted trajectory planning using blended near/mid/far offset proportional control |

## Results (10 episodes, seed 42, 2000 steps)

| Rank | Agent | Avg Reward | Best Episode | Completion % | Off-Track % |
|------|-------|-----------|-------------|-------------|-------------|
| 1st | Cautious (My Sister) | 630.8 | 875.6 | 70.7% | 16.7% |
| 2nd | Rain Rider | 623.3 | 877.2 | 69.8% | 19.6% |
| 3rd | MotoGP | 621.5 | 887.2 | 69.1% | 20.3% |
| 4th | Traction Focused | 609.2 | 864.4 | 68.2% | 19.1% |
| 5th | Line Hunter | 602.9 | 834.9 | 67.0% | 20.5% |
| 6th | Max Verstappen | 578.7 | 856.6 | 64.9% | 19.5% |
| 7th | Random Baseline | -129.2 | -118.7 | 0.0% | 14.0% |

All six rule-based agents beat the random baseline. The top agents achieve 65-70% average lap completion with best episodes reaching 95-98% completion (lap complete).

## Comparison Video

A side-by-side video of all agents running their best episodes is generated with:

```bash
python compare_video.py
```

This produces `results/videos/comparison.mp4` with all 7 agents in a grid, colour-coded borders, live HUD overlays, and "LAP COMPLETE" / "CRASHED" indicators when agents finish.

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `gymnasium[box2d]` requires [SWIG](https://www.swig.org/) to build Box2D. On Windows you may need `conda install swig` or download the SWIG binary.

## Usage

```bash
# Evaluate all agents (50 episodes each, headless)
python main.py

# Record video of the best episode per agent
python main.py --record

# Run a single agent with rendering
python main.py --agent motogp --render

# Quick test
python main.py --episodes 5

# Generate side-by-side comparison video
python compare_video.py

# Available agents: random, cautious, verstappen, traction, motogp, rain, linehunter
python main.py --agent rain --episodes 20
```

## What changed and why

### The Cursor starting point

The initial codebase was scaffolded using Cursor (AI-assisted IDE). Cursor built a working project structure with 5 agents, observation preprocessing, metrics, and visualisation. However, the agents performed poorly: all scored negative mean rewards (-48 to -63) with 59-73% off-track rates.

Cursor went through multiple debugging iterations:
- Fixed the zoom-in warmup animation (first 50 frames produce garbage features)
- Switched from ray-casting to track-center offset following
- Tuned steering cooldowns and thresholds multiple times
- Rebuilt the Superbike Pro agent from a voting system to a hierarchical priority system

Despite these improvements, **all agents scored negative rewards** because they spent 60-73% of their time off-track.

### The critical bug: inverted steering direction

The root cause was discovered through empirical testing. In CarRacing-v3, the camera rotates with the car (top-down, following), creating a counter-intuitive relationship between track offsets and steering:

- **Positive offset** (road center appears to the RIGHT) is corrected by steering **LEFT** (action 1)
- **Negative offset** (road center appears to the LEFT) is corrected by steering **RIGHT** (action 2)

All agents had this backwards. **Fixing this single bug improved scores from -60 average to +470 average**, a ~530-point improvement. Off-track rates dropped from 70-90% to 2-15%.

### What I got right

1. **Project structure**: clean separation of concerns (agents, environment, evaluation, config)
2. **Observation preprocessing**: grass-based track detection and track-center offset calculation
3. **Warmup handling**: correctly identified and handled the 50-frame zoom animation
4. **Reproducibility**: deterministic seeding per episode for fair comparison
5. **Visualisation suite**: 6 comparison plots generated automatically

### What was wrong

1. **Steering direction**: never empirically tested whether `LEFT if offset < 0` actually corrected a negative offset
2. **Over-engineering before validating basics**: built complex voting systems and state machines before verifying that basic steering worked
3. **Excessive cooldowns**: original cooldowns (4-12 steps) were far too long; the car needs to correct every 1-2 frames
4. **Conservative thresholds**: steering thresholds of 0.08-0.14 were too high for the small offsets on straight roads

### Improvements made after (help with AI)

1. **Fixed steering direction** in all agents
2. **Added two new agents**: Rain Rider (adaptive aggression) and Line Hunter (trajectory planning)
3. **Added five driving improvements** across all agents:
   - Off-track braking: brake when off-track at speed to prevent overshoot
   - Multi-frame recovery: commit to steering corrections for 3-5 frames instead of single-frame corrections
   - Curvature-based braking: use `abs(off_far - off_near)` as the primary brake trigger instead of just far offset
   - Proportional cooldowns: shorter cooldowns when far off-center for faster correction
   - Straight boost: gas harder when the road is straight (low curvature)
4. **Increased episode length** from 1000 to 2000 steps so agents can complete full laps
5. **Added comparison video** (`compare_video.py`) that shows all agents side-by-side
6. **Added video recording** (`--record`) with HUD overlay showing agent name, reward, speed, action, and off-track warnings

## Project Structure

```
CarRacing_AutonomousSystems/
├── README.md
├── requirements.txt
├── config.py                      # Shared constants and hyperparameters
├── main.py                        # Entry point: evaluate agents, generate plots
├── environment.py                 # Observation preprocessing (pixels → features)
│
├── race_game.py                   # Entry point wrapper → game/race.py
├── compare_video.py               # Entry point wrapper → evaluation/compare_video.py
├── generate_ghosts_batch.py       # Entry point wrapper → game/generate_ghosts.py
│
├── agents/                        # Rule-based agent implementations
│   ├── __init__.py                # Agent registry (ALL_AGENTS)
│   ├── base_agent.py              # Abstract base class
│   ├── baseline_random.py         # Baseline: random actions
│   ├── agent_cautious.py          # Cautious (My Sister)
│   ├── agent_apex.py              # Max Verstappen
│   ├── agent_traction.py          # Traction Focused
│   ├── agent_superbike.py         # MotoGP
│   ├── agent_rain.py              # Rain Rider
│   └── agent_line_hunter.py       # Line Hunter
│
├── game/                          # Championship race game (pygame)
│   ├── __init__.py
│   ├── race.py                    # Race loop, HUD, screens, ghost rendering
│   └── generate_ghosts.py         # Batch ghost data generation (subprocess)
│
├── evaluation/                    # Metrics, plots, and video comparison
│   ├── __init__.py
│   ├── metrics.py                 # Per-episode metric tracking
│   ├── visualize.py               # matplotlib/seaborn comparison plots
│   └── compare_video.py           # Side-by-side agent comparison video
│
└── results/                       # Auto-generated outputs
    ├── results.csv
    ├── ghosts/                    # Ghost replay data (JSON per agent per seed)
    └── videos/
        ├── comparison.mp4         # All agents side-by-side
        └── best_*.mp4             # Individual best-episode recordings
```

## Metrics

Each agent is evaluated on:

- **Total reward**: primary environment score (+1000/N per tile visited, -0.1 per frame)
- **Lap completion %**: approximated from cumulative reward (reward / 900 * 100)
- **Survival steps**: how long the agent lasts (max 2000)
- **Off-track %**: proportion of steps spent on grass
- **Average speed**: estimated from the on-screen speed bar and frame differencing

## Observation Preprocessing

The `environment.py` module converts each 96x96x3 RGB frame into a feature dictionary:

| Feature | Description |
|---|---|
| `offset_near` | Track center offset at 10 rows ahead [-1, 1] |
| `offset_mid` | Track center offset at 25 rows ahead [-1, 1] |
| `offset_far` | Track center offset at 40 rows ahead [-1, 1] |
| `curve_direction` | -1 (left), 0 (straight), +1 (right) |
| `curve_sharpness` | 0.0 (straight) to 1.0 (hairpin) |
| `speed` | Estimated speed [0, 1] |
| `on_track` | Boolean: is the car on the road? |
| `warmup` | True for first 50 frames (zoom animation) |

Track detection uses grass colour isolation (green channel dominant), and the road is defined as everything that is NOT grass with minimum brightness > 40.

## Reproducibility

All experiments use a deterministic base seed (default `42`). Each episode `i` receives seed `base_seed + i`, ensuring identical track layouts across agents.

