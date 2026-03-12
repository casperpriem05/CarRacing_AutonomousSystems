"""Agent 3 -- Traction Manager.

Motorcycle metaphor: a rider focused on grip management and smooth inputs,
inspired by the *friction circle* in motorcycle dynamics.

Priority order: Stability > Completion > Speed.

Uses track offsets combined with a traction budget.  Steering at high
speed costs more budget.  If budget is depleted, the agent coasts to
regenerate.  Lower thresholds and shorter cooldowns than previous version.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np

import config
from .base_agent import BaseAgent

DO_NOTHING = 0
LEFT = 1
RIGHT = 2
GAS = 3
BRAKE = 4

HISTORY_LEN = 8


class TractionManager(BaseAgent):
    """Grip-aware rider -- never exceeds the friction circle."""

    def __init__(self) -> None:
        super().__init__(name="Traction Focused")
        self._budget = config.TRACTION_BUDGET_MAX
        self._history: deque[int] = deque(maxlen=HISTORY_LEN)
        self._steer_cooldown: int = 0
        self._recovery_frames: int = 0
        self._recovery_dir: int = DO_NOTHING

    def reset(self) -> None:
        self._budget = config.TRACTION_BUDGET_MAX
        self._history.clear()
        self._steer_cooldown = 0
        self._recovery_frames = 0
        self._recovery_dir = DO_NOTHING

    def _consume(self, amount: float) -> None:
        self._budget = float(np.clip(self._budget - amount, 0.0, config.TRACTION_BUDGET_MAX))

    def _regenerate(self, amount: float = 0.12) -> None:
        self._budget = float(np.clip(self._budget + amount, 0.0, config.TRACTION_BUDGET_MAX))

    def _would_oscillate(self, action: int) -> bool:
        if len(self._history) < 2:
            return False
        last = self._history[-1]
        return (action == LEFT and last == RIGHT) or (action == RIGHT and last == LEFT)

    def _record(self, action: int) -> int:
        self._history.append(action)
        return action

    def _steer_toward(self, offset: float) -> int:
        return RIGHT if offset < 0 else LEFT

    def act(self, features: Dict[str, Any]) -> int:
        if features.get("warmup", False):
            return self._record(GAS)

        if self._steer_cooldown > 0:
            self._steer_cooldown -= 1

        speed = features["speed"]
        on_track = features["on_track"]
        off_near = features.get("offset_near", 0.0)
        off_mid = features.get("offset_mid", 0.0)
        off_far = features.get("offset_far", 0.0)
        curvature = abs(off_far - off_near)

        self._regenerate()

        # Recovery mode
        if self._recovery_frames > 0:
            self._recovery_frames -= 1
            if on_track and abs(off_near) < 0.04:
                self._recovery_frames = 0
            else:
                return self._record(self._recovery_dir)

        # Off-track emergency
        if not on_track:
            self._consume(0.10)
            # Brake first if fast to avoid overshoot
            if speed > 0.30:
                return self._record(BRAKE)
            # Multi-frame recovery commitment
            if abs(off_near) > 0.03:
                self._recovery_dir = self._steer_toward(off_near)
                self._recovery_frames = 4
                return self._record(self._recovery_dir)
            if abs(off_mid) > 0.03:
                self._recovery_dir = self._steer_toward(off_mid)
                self._recovery_frames = 3
                return self._record(self._recovery_dir)
            return self._record(GAS)

        if speed < 0.03:
            return self._record(GAS)

        # Steering with traction budget
        if abs(off_near) > 0.05 and self._steer_cooldown == 0:
            steer_cost = 0.06 + speed * 0.08
            steer_action = self._steer_toward(off_near)

            if self._would_oscillate(steer_action):
                return self._record(GAS)

            if self._budget >= steer_cost:
                self._consume(steer_cost)
                # Proportional cooldown: bigger offset = shorter cooldown
                if abs(off_near) > 0.15:
                    self._steer_cooldown = 1
                elif abs(off_near) > 0.08:
                    self._steer_cooldown = 1
                else:
                    self._steer_cooldown = max(1, int(speed * 2))
                return self._record(steer_action)

        # Anticipatory from mid offset
        if abs(off_mid) > 0.10 and abs(off_near) < 0.05 and self._steer_cooldown == 0:
            steer = self._steer_toward(off_mid)
            if not self._would_oscillate(steer):
                self._consume(0.04)
                self._steer_cooldown = 2
                return self._record(steer)

        # Braking
        if speed > 0.35 and curvature > 0.18:
            self._consume(0.08)
            return self._record(BRAKE)
        if speed > 0.35 and abs(off_far) > 0.22:
            self._consume(0.08)
            return self._record(BRAKE)

        if self._budget < 0.15:
            return self._record(DO_NOTHING)

        # Straight boost
        if curvature < 0.05 and abs(off_near) < 0.08:
            return self._record(GAS)

        # Coast approaching curves at moderate speed
        if speed > 0.40 and curvature > 0.08:
            return self._record(DO_NOTHING)

        return self._record(GAS)
