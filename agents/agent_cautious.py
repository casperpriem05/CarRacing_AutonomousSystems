"""Agent 1 -- Cautious Cruiser.

Motorcycle metaphor: a new rider who prioritises safety above all else.

Priority order: Safety > Completion > Speed.

Uses track-center offsets for gentle, frequent steering corrections.
Defaults to gas whenever possible.  Brakes early if the road shifts
sharply ahead.  Uses anti-oscillation instead of time-based cooldowns.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import config
from .base_agent import BaseAgent

DO_NOTHING = 0
LEFT = 1
RIGHT = 2
GAS = 3
BRAKE = 4


class CautiousCruiser(BaseAgent):
    """Safety-first rider -- brakes early, accelerates gently."""

    def __init__(self) -> None:
        super().__init__(name="Cautious (My Sister)")
        self._last_actions: deque[int] = deque(maxlen=4)
        self._steer_cooldown: int = 0
        self._recovery_frames: int = 0
        self._recovery_dir: int = DO_NOTHING

    def reset(self) -> None:
        self._last_actions.clear()
        self._steer_cooldown = 0
        self._recovery_frames = 0
        self._recovery_dir = DO_NOTHING

    def _would_oscillate(self, action: int) -> bool:
        if len(self._last_actions) < 2:
            return False
        prev = self._last_actions[-1]
        return (action == LEFT and prev == RIGHT) or (action == RIGHT and prev == LEFT)

    def _pick(self, action: int) -> int:
        self._last_actions.append(action)
        return action

    def _steer_toward(self, offset: float) -> int:
        return RIGHT if offset < 0 else LEFT

    def act(self, features: Dict[str, Any]) -> int:
        if features.get("warmup", False):
            return self._pick(GAS)

        if self._steer_cooldown > 0:
            self._steer_cooldown -= 1

        speed = features["speed"]
        on_track = features["on_track"]
        off_near = features.get("offset_near", 0.0)
        off_mid = features.get("offset_mid", 0.0)
        off_far = features.get("offset_far", 0.0)
        curvature = abs(off_far - off_near)

        # Recovery mode
        if self._recovery_frames > 0:
            self._recovery_frames -= 1
            if on_track and abs(off_near) < 0.04:
                self._recovery_frames = 0  # back on track, cancel recovery
            else:
                return self._pick(self._recovery_dir)

        # Off-track emergency
        if not on_track:
            # Brake first if going fast to avoid overshoot
            if speed > 0.30:
                return self._pick(BRAKE)
            # Commit to multi-frame recovery
            if abs(off_near) > 0.03:
                self._recovery_dir = self._steer_toward(off_near)
                self._recovery_frames = 4
                return self._pick(self._recovery_dir)
            if abs(off_mid) > 0.03:
                self._recovery_dir = self._steer_toward(off_mid)
                self._recovery_frames = 3
                return self._pick(self._recovery_dir)
            return self._pick(GAS)

        if speed < 0.03:
            return self._pick(GAS)

        # Braking
        if curvature > 0.20 and speed > config.CAUTIOUS_MAX_SPEED:
            return self._pick(BRAKE)
        if abs(off_far) > 0.22 and speed > config.CAUTIOUS_MAX_SPEED:
            return self._pick(BRAKE)

        # Steering
        if abs(off_near) > 0.05 and self._steer_cooldown == 0:
            steer = self._steer_toward(off_near)
            if self._would_oscillate(steer):
                return self._pick(GAS)
            # Proportional: bigger offset = shorter cooldown
            if abs(off_near) > 0.15:
                self._steer_cooldown = 1
            elif abs(off_near) > 0.08:
                self._steer_cooldown = 1
            else:
                self._steer_cooldown = 2
            return self._pick(steer)

        # Anticipatory correction from mid offset
        if abs(off_mid) > 0.10 and abs(off_near) < 0.05 and self._steer_cooldown == 0:
            steer = self._steer_toward(off_mid)
            if not self._would_oscillate(steer):
                self._steer_cooldown = 2
                return self._pick(steer)

        # Straight boost
        if curvature < 0.05 and abs(off_near) < 0.08:
            return self._pick(GAS)

        # Coast if approaching a curve at moderate speed
        if speed > 0.40 and curvature > 0.08:
            return self._pick(DO_NOTHING)

        return self._pick(GAS)
