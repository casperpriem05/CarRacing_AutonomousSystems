"""Shared constants and hyperparameters for the CarRacing rule-based agents project."""

NUM_EPISODES: int = 50
BASE_SEED: int = 42

ENV_ID: str = "CarRacing-v3"
MAX_EPISODE_STEPS: int = 2000

CAUTIOUS_MAX_SPEED: float = 0.45
TRACTION_BUDGET_MAX: float = 1.0

RESULTS_DIR: str = "results"
RESULTS_CSV: str = "results/results.csv"
VIDEO_DIR: str = "results/videos"
VIDEO_FPS: int = 30
