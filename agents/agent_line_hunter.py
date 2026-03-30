"""Agent 6 -- Line Hunter.

Motorcycle metaphor: a precision rider who obsessively follows the
optimal racing line, using all three offset distances to plan a smooth
trajectory through every corner.

Architecture: weighted trajectory planning with proportional control.

    - Computes a 'trajectory error' from near/mid/far offsets
    - Applies proportional corrections scaled to error magnitude
    - Uses trajectory curvature to decide brake vs steer
    - Proportional cooldowns allow faster correction when far off-center

Key advantages:
    * Smoothest trajectory of all agents -> fewer jerky corrections
    * Uses curvature (far-near divergence) for precise brake decisions
    * Proportional error scaling avoids over/under-correction
    * Off-track recovery: brakes then commits to multi-frame correction
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np

from .base_agent import BaseAgent

DO_NOTHING = 0
LEFT = 1
RIGHT = 2
GAS = 3
BRAKE = 4


class LineHunter(BaseAgent):
    """Precision line-follower using trajectory planning."""

    def __init__(self) -> None:
        super().__init__(name="Line Hunter")
        self._last_actions: deque[int] = deque(maxlen=6)
        self._steer_cooldown: int = 0
        self._consecutive_steer: int = 0
        self._recovery_frames: int = 0
        self._recovery_dir: int = DO_NOTHING

    def reset(self) -> None:
        self._last_actions.clear()
        self._steer_cooldown = 0
        self._consecutive_steer = 0
        self._recovery_frames = 0
        self._recovery_dir = DO_NOTHING

    def _would_oscillate(self, action: int) -> bool:
        if len(self._last_actions) < 2:
            return False
        prev = self._last_actions[-1]
        return (action == LEFT and prev == RIGHT) or (action == RIGHT and prev == LEFT)

    def _pick(self, action: int) -> int:
        self._last_actions.append(action)
        if action in (LEFT, RIGHT):
            self._consecutive_steer += 1
        else:
            self._consecutive_steer = 0
        return action

    def _steer_toward(self, offset: float) -> int:
        return RIGHT if offset < 0 else LEFT

    def act(self, features: Dict[str, Any]) -> int:
        if features.get("warmup", False):
            return self._pick(GAS)

        if self._steer_cooldown > 0:
            self._steer_cooldown -= 1

        speed: float = features["speed"]
        on_track: bool = features["on_track"]
        off_near: float = features.get("offset_near", 0.0)
        off_mid: float = features.get("offset_mid", 0.0)
        off_far: float = features.get("offset_far", 0.0)

        # Trajectory error: weighted blend of all 3 distances
        trajectory_error = off_near * 0.5 + off_mid * 0.3 + off_far * 0.2
        curvature = abs(off_far - off_near)

        # Recovery mode
        if self._recovery_frames > 0:
            self._recovery_frames -= 1
            if on_track and abs(off_near) < 0.04:
                self._recovery_frames = 0
            else:
                return self._pick(self._recovery_dir)

        # Off-track emergency
        if not on_track:
            # Brake first if fast to prevent overshoot
            if speed > 0.30:
                return self._pick(BRAKE)
            # Multi-frame recovery commitment
            if abs(off_near) > 0.03:
                self._recovery_dir = self._steer_toward(off_near)
                self._recovery_frames = 4
                return self._pick(self._recovery_dir)
            if abs(off_mid) > 0.03:
                self._recovery_dir = self._steer_toward(off_mid)
                self._recovery_frames = 3
                return self._pick(self._recovery_dir)
            return self._pick(GAS)

        # Braking
        if curvature > 0.18 and speed > 0.30:
            return self._pick(BRAKE)
        if abs(off_far) > 0.22 and speed > 0.35:
            return self._pick(BRAKE)

        # Steering
        if self._steer_cooldown == 0:
            error_mag = abs(trajectory_error)

            if error_mag > 0.025:
                steer = self._steer_toward(trajectory_error)

                if not self._would_oscillate(steer):
                    cooldown = 1

                    # Prevent excessive same-direction steering
                    if self._consecutive_steer > 8:
                        cooldown = 2

                    self._steer_cooldown = cooldown
                    return self._pick(steer)

        # Speed management
        # Straight boost: gas hard when road is straight
        if curvature < 0.05 and abs(off_near) < 0.08:
            return self._pick(GAS)

        if speed > 0.45 and curvature > 0.08:
            return self._pick(DO_NOTHING)

        return self._pick(GAS)
